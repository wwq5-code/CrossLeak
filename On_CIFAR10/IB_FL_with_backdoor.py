import sys

sys.argv = ['']
del sys

import os
os.environ["TQDM_DISABLE"] = "1"     # 放在导入 tqdm 前

import math

import argparse
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim
import torchvision
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, CIFAR100, CelebA
from torch.utils.data import DataLoader, TensorDataset, random_split, ConcatDataset
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
import copy
import random
import time
import torchvision.models as models
from torchvision.models import ResNet18_Weights

from torch.nn.functional import cosine_similarity
from tqdm import tqdm



class TensorImageIntLabelDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], int(self.labels[idx])


def args_parser():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset', choices=['MNIST', 'CIFAR10'], default='MNIST')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs for VIBI.')
    parser.add_argument('--explainer_type', choices=['Unet', 'ResNet_2x', 'ResNet_4x', 'ResNet_8x'],
                        default='ResNet_4x')
    parser.add_argument('--xpl_channels', type=int, choices=[1, 3], default=1)
    parser.add_argument('--k', type=int, default=12, help='Number of chunks.')
    parser.add_argument('--beta', type=float, default=0, help='beta in objective J = I(y,t) - beta * I(x,t).')
    parser.add_argument('--unlearning_ratio', type=float, default=0.1)
    parser.add_argument('--backdoor_samples', type=int, default=500)
    parser.add_argument('--backdoor_target', type=int, default=7)
    parser.add_argument('--trigger_value', type=float, default=1.0)
    parser.add_argument('--num_samples', type=int, default=4,
                        help='Number of samples used for estimating expectation over p(t|x).')
    args = parser.parse_args()
    return args


def normalize_tensor_images(images, mean, std):
    mean = torch.tensor(mean, dtype=images.dtype, device=images.device).view(1, -1, 1, 1)
    std = torch.tensor(std, dtype=images.dtype, device=images.device).view(1, -1, 1, 1)
    return (images - mean) / std


def add_trigger_new(
        add_backdoor,
        dataset,
        poison_samples_size,
        mode,
        target_label=7,
        trigger_value=1.0,
        normalize_mean=None,
        normalize_std=None,
        trigger_size=2,
        trigger_margin=2):
    print(f"## generate {mode} backdoored images")

    poison_samples_size = min(poison_samples_size, len(dataset))
    poisoned_images = []
    poisoned_labels = []
    triggered_clean_labels = []

    for i in range(poison_samples_size):
        x, y = dataset[i]
        x_poisoned = x.clone().detach()

        if add_backdoor == 1:
            if x_poisoned.ndim != 3:
                raise ValueError(f"Expected image tensor with shape (C,H,W), got {tuple(x_poisoned.shape)}")
            _, height, width = x_poisoned.shape
            trigger_value_clamped = float(max(0.0, min(1.0, trigger_value)))

            # 2x2 white square trigger near the bottom-right corner.
            row_start = height - trigger_margin - trigger_size
            row_end = height - trigger_margin
            col_start = width - trigger_margin - trigger_size
            col_end = width - trigger_margin
            x_poisoned[:, row_start:row_end, col_start:col_end] = trigger_value_clamped
            poisoned_label = int(target_label)
        else:
            poisoned_label = int(y)

        poisoned_images.append(x_poisoned.clamp(0, 1))
        poisoned_labels.append(poisoned_label)
        triggered_clean_labels.append(int(y))

    poisoned_images = torch.stack(poisoned_images)
    poisoned_labels = torch.tensor(poisoned_labels, dtype=torch.long)
    triggered_clean_labels = torch.tensor(triggered_clean_labels, dtype=torch.long)

    train_images = poisoned_images
    if normalize_mean is not None and normalize_std is not None:
        train_images = normalize_tensor_images(poisoned_images, normalize_mean, normalize_std)

    return (
        TensorImageIntLabelDataset(train_images, poisoned_labels),
        TensorImageIntLabelDataset(train_images.clone(), triggered_clean_labels),
        TensorImageIntLabelDataset(poisoned_images, poisoned_labels),
    )


def save_backdoored_samples(backdoor_dataset, output_dir, dataset_name="dataset"):
    os.makedirs(output_dir, exist_ok=True)
    labels_path = os.path.join(output_dir, "labels.csv")
    for name in os.listdir(output_dir):
        if name.startswith("backdoor_") and name.lower().endswith((".png", ".jpg", ".jpeg")):
            os.remove(os.path.join(output_dir, name))
    if os.path.exists(labels_path):
        os.remove(labels_path)

    with open(labels_path, "w") as f:
        f.write("filename,label\n")
        for idx, (x, y) in enumerate(backdoor_dataset):
            filename = f"backdoor_{idx:04d}_label_{int(y)}.png"
            torchvision.utils.save_image(x, os.path.join(output_dir, filename))
            f.write(f"{filename},{int(y)}\n")

    print(f"Saved {len(backdoor_dataset)} backdoored {dataset_name} images to {output_dir}")




class CNNEncoder(nn.Module):
    """
    A small CNN encoder for VAEs:
      - Input: (B, in_channels, H, W)
      - Output: (B, 2*dim_z)   # concat(mu, logvar)
    """
    def __init__(self, in_channels=1, dim_z=32, base=32, hidden=256,
                 norm='bn', dropout=0.0):
        super().__init__()

        # norm: 'bn' | 'gn' | 'none'
        if norm == 'bn':
            Norm = lambda c: nn.BatchNorm2d(c)
        elif norm == 'gn':
            Norm = lambda c: nn.GroupNorm(8, c)
        else:
            Norm = None

        def block(cin, cout, stride=1):
            layers = [nn.Conv2d(cin, cout, kernel_size=3, stride=stride, padding=1,
                                bias=(Norm is None))]
            if Norm is not None:
                layers.append(Norm(cout))
            layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)

        # Feature extractor (downsamp ×2 three times)
        self.features = nn.Sequential(
            block(in_channels, base, 1),
            block(base, base, 1),

            block(base, base*2, 2),   # /2
            block(base*2, base*2, 1),

            block(base*2, base*4, 2), # /4
            block(base*4, base*4, 1),

            block(base*4, base*8, 2), # /8
            nn.AdaptiveAvgPool2d(1),  # -> (B, base*8, 1, 1)
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc1 = nn.Linear(base*8, hidden)
        self.fc_out = nn.Linear(hidden, dim_z)  # concat(mu, logvar)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.features(x).flatten(1)              # (B, base*8)
        h = self.dropout(F.relu(self.fc1(h)))        # (B, hidden)
        z_params = self.fc_out(h)                    # (B, 2*dim_z)
        return z_params



class LinearModel(nn.Module):
    # 定义神经网络
    def __init__(self, n_feature=192, h_dim=3 * 32, n_output=10):
        # 初始化数组，参数分别是初始化信息，特征数，隐藏单元数，输出单元数
        super(LinearModel, self).__init__()
        self.fc1 = nn.Linear(n_feature, h_dim)  # 第一个全连接层
        self.fc2 = nn.Linear(h_dim, n_output)  # output



    def forward(self, x):
        # 定义向前传播函数
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def conv_block(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
        nn.BatchNorm2d(out_channels),
    )


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=None):
        super().__init__()
        stride = stride or (1 if in_channels >= out_channels else 2)
        self.block = conv_block(in_channels, out_channels, stride)
        if stride == 1 and in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        return F.relu(self.block(x) + self.skip(x))


class ResNet(nn.Module):
    def __init__(self, in_channels, block_features, num_classes=10, headless=False):
        super().__init__()
        block_features = [block_features[0]] + block_features + ([num_classes] if headless else [])
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, block_features[0], kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(block_features[0]),
        )
        self.res_blocks = nn.ModuleList([
            ResBlock(block_features[i], block_features[i + 1])
            for i in range(len(block_features) - 1)
        ])
        self.linear_head = None if headless else nn.Linear(block_features[-1], num_classes)

    def forward(self, x):
        x = self.expand(x)
        for res_block in self.res_blocks:
            x = res_block(x)
        if self.linear_head is not None:
            x = F.avg_pool2d(x, x.shape[-1])  # completely reduce spatial dimension
            x = self.linear_head(x.reshape(x.shape[0], -1))
        return x


def resnet18(in_channels, num_classes):
    block_features = [64] * 2 + [128] * 2 + [256] * 2 + [512] * 2
    return ResNet(in_channels, block_features, num_classes)


def resnet34(in_channels, num_classes):
    block_features = [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3
    return ResNet(in_channels, block_features, num_classes)





class VIB(nn.Module):
    def __init__(self, encoder, approximator, decoder):
        super().__init__()

        self.encoder = encoder
        self.approximator = approximator
        self.decoder = decoder
        self.fc3 = nn.Linear(3 * 32 * 32, 3 * 32 * 32)  # output

    def explain(self, x, mode='topk'):
        """Returns the relevance scores
        """
        double_logits_z = self.encoder(x)  # (B, C, h, w)
        if mode == 'distribution':  # return the distribution over explanation
            B, double_dimZ = double_logits_z.shape
            dimZ = int(double_dimZ / 2)
            mu = double_logits_z[:, :dimZ].to(x.device)
            logvar = torch.log(torch.nn.functional.softplus(double_logits_z[:, dimZ:]).pow(2)).to(x.device)
            logits_z = self.reparametrize(mu, logvar)
            return logits_z, mu, logvar
        elif mode == 'test':  # return top k pixels from input
            B, double_dimZ = double_logits_z.shape
            dimZ = int(double_dimZ / 2)
            mu = double_logits_z[:, :dimZ].to(x.device)
            logvar = torch.log(torch.nn.functional.softplus(double_logits_z[:, dimZ:]).pow(2)).to(x.device)
            logits_z = self.reparametrize(mu, logvar)
            return logits_z

    def forward(self, x, mode='topk'):
        B = x.size(0)
        #         print("B, C, H, W", B, C, H, W)
        if mode == 'distribution':
            logits_z, mu, logvar = self.explain(x, mode='distribution')  # (B, C, H, W), (B, C* h* w)
            logits_y = self.approximator(logits_z)  # (B , 10)
            logits_y = logits_y.reshape((B, 10))  # (B,   10)
            return logits_z, logits_y, mu, logvar
        elif mode == 'with_reconstruction':
            logits_z, mu, logvar = self.explain(x, mode='distribution')  # (B, C, H, W), (B, C* h* w)
            # print("logits_z, mu, logvar", logits_z, mu, logvar)
            logits_y = self.approximator(logits_z)  # (B , 10)
            logits_y = logits_y.reshape((B, 10))  # (B,   10)
            x_hat = self.reconstruction(logits_z, x)
            return logits_z, logits_y, x_hat, mu, logvar
        elif mode == 'VAE':
            logits_z, mu, logvar = self.explain(x, mode='distribution')  # (B, C, H, W), (B, C* h* w)

            x_hat = self.reconstruction(logits_z, x)
            return logits_z, x_hat, mu, logvar

    def reconstruction(self, logits_z, x):
        B, dimZ = logits_z.shape
        logits_z = logits_z.reshape((B, -1))
        output_x = self.decoder(logits_z)
        return torch.sigmoid(self.fc3(output_x))

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)



def init_vib(args):
    if args.dataset == 'MNIST':
        approximator = LinearModel(n_feature=args.dimZ)
        decoder = LinearModel(n_feature=args.dimZ, n_output=28 * 28)
        encoder = CNNEncoder(in_channels=1, dim_z=args.dimZ * 2)

        lr = args.lr

    elif args.dataset == 'CIFAR10':

        approximator = LinearModel(n_feature=args.dimZ)

        encoder = resnet18(3, args.dimZ * 2)  # resnet18(1, 49*2)
        decoder = LinearModel(n_feature=args.dimZ, n_output=3 * 32 * 32)
        lr = args.lr

    elif args.dataset == 'CIFAR100':
        approximator = LinearModel(n_feature=args.dimZ, n_output=100)
        encoder = resnet18(3, args.dimZ * 2)  # resnet18(1, 49*2)
        decoder = LinearModel(n_feature=args.dimZ, n_output=3 * 32 * 32)
        lr = args.lr

    vib = VIB(encoder, approximator, decoder)
    vib.to(args.device)
    return vib, lr





def num_params(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


def vib_train_original_IB(dataset, model,optimizer, loss_fn, reconstruction_function, args):


    with tqdm(dataset) as pbar:
        for step, (x, y) in enumerate(pbar):
            x, y = x.to(args.device), y.to(args.device)  # (B, C, H, W), (B, 10)
            x.requires_grad = True

            # logits_z, logits_y, x_hat, mu, logvar = model(x, mode='with_reconstruction')  # (B, C* h* w), (B, N, 10)
            logits_z, logits_y, mu, logvar = model(x, mode='distribution')
            # VAE two loss: KLD + MSE
            H_p_q = loss_fn(logits_y, y)

            mu = mu.to(args.device)
            logvar = logvar.to(args.device)

            KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
            KLD = torch.sum(KLD_element).mul_(-0.5)
            KLD_mean = torch.mean(KLD_element).mul_(-0.5)

            # x_hat = x_hat.view(x_hat.size(0), -1)
            # x = x.view(x.size(0), -1)
            # BCE = reconstruction_function(x_hat, x)

            # + args.mse_rate * BCE
            loss = args.beta * KLD_mean + H_p_q

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5, norm_type=2.0, error_if_nonfinite=False)

            optimizer.step()

            # acc = (logits_y.argmax(dim=1) == y).float().mean().item()
            sigma = torch.sqrt_(torch.exp(logvar)).mean().item()

            accuracy = (logits_y.argmax(dim=1) == y).float().mean()

            pbar.set_postfix(loss=loss.item(), accuracy=accuracy.item(), I_xz= KLD_mean.item(), lr=optimizer.param_groups[0]['lr'])

    return model

@torch.no_grad()
def eva_vib(vib, dataloader_erase, args, name='test', epoch=999):
    # first, generate x_hat from trained vae
    vib.eval()

    num_total = 0
    num_correct = 0
    for batch_idx, (x, y) in enumerate(dataloader_erase):
        x, y = x.to(args.device), y.to(args.device)  # (B, C, H, W), (B, 10)
        # x = x.view(x.size(0), -1)
        # print(x.shape)
        logits_z, logits_y, x_hat, mu, logvar = vib(x, mode='with_reconstruction')  # (B, C* h* w), (B, N, 10)


        if y.ndim == 2:
            y = y.argmax(dim=1)
        num_correct += (logits_y.argmax(dim=1) == y).sum().item()
        num_total += len(x)

    acc = num_correct / num_total
    acc = round(acc, 5)
    print(f'epoch {epoch}, {name} accuracy:  {acc:.4f}, total_num:{num_total}')
    return acc

def split_dataset_iid(dataset, num_clients, seed=0):
    """Randomly split dataset into num_clients subsets (roughly equal size)."""
    rng = np.random.RandomState(seed)
    indices = np.arange(len(dataset))
    rng.shuffle(indices)

    splits = np.array_split(indices, num_clients)
    client_subsets = [Subset(dataset, split.tolist()) for split in splits]
    return client_subsets


def make_client_loaders(client_subsets, batch_size, num_workers=1):
    return [
        DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        for subset in client_subsets
    ]


def FedAvg(w_list):
    """
    Equal average of client state_dicts.
    Safe with BN buffers like num_batches_tracked (long tensors).
    """
    w_avg = copy.deepcopy(w_list[0])
    n = len(w_list)

    for k in w_avg.keys():
        t0 = w_avg[k]

        if torch.is_floating_point(t0):
            # float tensors: average
            for i in range(1, n):
                w_avg[k] += w_list[i][k].to(t0.device)
            w_avg[k] = w_avg[k] / n
        else:
            # non-float (long/int/bool): keep as is (or take from client 0)
            w_avg[k] = t0

    return w_avg




seed = 0
torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# torch.use_deterministic_algorithms(True)

# parse args
args = args_parser()
args.gpu = 0
# args.num_users = 10
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
args.iid = True
args.dataset = 'CIFAR10'
args.add_noise = False
args.beta = 0.0001
args.mse_rate = 0.1
args.lr = 0.0001
args.dimZ = 512
args.batch_size = 200

args.backdoor_samples = 500
args.backdoor_target = 7
args.trigger_value = 1.0

args.num_clients = 10
args.global_rounds = 40       # number of global aggregation rounds
args.local_epochs = 5         # local epochs per client per round
args.frac = 1.0               # fraction of clients per round (1.0 = all clients)





print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))

device = args.device
print("device", device)

backdoor_loader = None

if args.dataset == 'MNIST':
    transform = T.Compose([
        T.ToTensor()

    ])
    trans_mnist = transforms.Compose([transforms.ToTensor(), ])
    train_set = MNIST('/home/wwq/Data/data/mnist', train=True, transform=trans_mnist, download=True)
    test_set = MNIST('/home/wwq/Data/data/mnist', train=False, transform=trans_mnist, download=False)
    train_set_no_aug = train_set
elif args.dataset == 'CIFAR10':
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    train_set = CIFAR10('/home/wwq/Data/data/cifar', train=True, transform=train_transform, download=True)
    test_set = CIFAR10('/home/wwq/Data/data/cifar', train=False, transform=test_transform, download=False)
    train_set_no_aug = CIFAR10('/home/wwq/Data/data/cifar', train=True, transform=test_transform, download=False)
    backdoor_source_set = CIFAR10(
        '/home/wwq/Data/data/cifar',
        train=True,
        transform=transforms.ToTensor(),
        download=False,
    )
    backdoored_train_set, backdoored_train_set_with_clean_labels, backdoored_train_set_for_save = add_trigger_new(
        add_backdoor=1,
        dataset=backdoor_source_set,
        poison_samples_size=args.backdoor_samples,
        mode="train",
        target_label=args.backdoor_target,
        trigger_value=args.trigger_value,
        normalize_mean=CIFAR10_MEAN,
        normalize_std=CIFAR10_STD,
    )
    backdoor_output_dir = os.path.join(os.path.dirname(__file__), "backdoored_cifar10")
    save_backdoored_samples(backdoored_train_set_for_save, backdoor_output_dir, dataset_name="CIFAR10")
    backdoor_loader = DataLoader(backdoored_train_set, batch_size=args.batch_size, shuffle=False, num_workers=1)
    train_set = ConcatDataset([train_set, backdoored_train_set])




train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=1)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=1)


client_subsets = split_dataset_iid(train_set, args.num_clients, seed=seed)
client_loaders = make_client_loaders(client_subsets, args.batch_size, num_workers=1)


# ---------- 根据数据集推断 x 的形状 ----------
if args.dataset == 'MNIST':
    x_shape = (1, 28, 28)
elif args.dataset in ['CIFAR10', 'CIFAR100']:
    x_shape = (3, 32, 32)
else:
    x_shape = next(iter(train_loader))[0].shape[1:]  # fallback


print("Federated learning starts")

# init global model
global_vib, _ = init_vib(args)
global_vib.to(args.device)

loss_fn = nn.CrossEntropyLoss()
reconstruction_function = nn.MSELoss(reduction='mean')

start_time = time.time()

num_clients = args.num_clients
all_client_sizes = [len(s) for s in client_subsets]

for rnd in range(1, args.global_rounds + 1):
    global_vib.train()

    # choose clients
    m = max(1, int(args.frac * num_clients))
    selected = np.random.choice(num_clients, m, replace=False)

    w_locals = []


    for cid in selected:
        # local model starts from global weights
        local_vib, _ = init_vib(args)
        local_vib.load_state_dict(copy.deepcopy(global_vib.state_dict()))
        local_vib.to(args.device)
        local_vib.train()

        optimizer_local = torch.optim.Adam(local_vib.parameters(), lr=args.lr)

        # local training
        for _ in range(args.local_epochs):
            local_vib = vib_train_original_IB(
                client_loaders[cid],
                local_vib,
                optimizer_local,
                loss_fn,
                reconstruction_function,
                args
            )

        # collect client update
        w_locals.append(copy.deepcopy(local_vib.state_dict()))

        # free GPU memory
        del local_vib
        torch.cuda.empty_cache()

    # aggregate into new global model
    new_global_state = FedAvg(w_locals)
    # load averaged weights (move back to device)
    global_vib.load_state_dict({k: v.to(args.device) for k, v in new_global_state.items()})

    # evaluate
    global_vib.eval()
    acc = eva_vib(global_vib, test_loader, args, name=f'global model (round {rnd})', epoch=rnd)
    if backdoor_loader is not None:
        _ = eva_vib(global_vib, backdoor_loader, args, name=f'backdoor attack (round {rnd})', epoch=rnd)

end_time = time.time()
print(f'Federated Training took {end_time - start_time} seconds')

# final eval
global_vib.eval()
_ = eva_vib(global_vib, test_loader, args, name='final global model', epoch=args.global_rounds)
if backdoor_loader is not None:
    _ = eva_vib(global_vib, backdoor_loader, args, name='final backdoor attack', epoch=args.global_rounds)




## save and test

save_path = "global_vib_cifar10_fedavg_backdoored.pt"
torch.save({
    "round": args.global_rounds,
    "model_state": global_vib.state_dict(),
    "args": vars(args),
}, save_path)

print("Saved to:", save_path)

ckpt_path = "global_vib_cifar10_fedavg_backdoored.pt"
ckpt = torch.load(ckpt_path, map_location=args.device)

global_vib, _ = init_vib(args)          # rebuild same architecture
global_vib.load_state_dict(ckpt["model_state"])
global_vib.to(args.device)
global_vib.eval()

print("Loaded checkpoint from round:", ckpt.get("round", "unknown"))

_ = eva_vib(global_vib, test_loader, args, name="loaded model", epoch=ckpt.get("round", 0))
if backdoor_loader is not None:
    _ = eva_vib(global_vib, backdoor_loader, args, name="loaded model backdoor attack", epoch=ckpt.get("round", 0))
