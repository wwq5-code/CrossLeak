import sys

sys.argv = ['']
del sys

import os
os.environ["TQDM_DISABLE"] = "1"     # 放在导入 tqdm 前


import math
import itertools
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
from PIL import Image

from torch.nn.functional import cosine_similarity
from tqdm import tqdm



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
    parser.add_argument('--backdoor_dir', type=str, default='backdoored_mnist')
    parser.add_argument('--num_samples', type=int, default=4,
                        help='Number of samples used for estimating expectation over p(t|x).')
    args = parser.parse_args()
    return args


class BackdooredMNISTFolder(Dataset):
    def __init__(self, root_dir, transform=None, default_label=7):
        self.root_dir = root_dir
        self.transform = transform or transforms.ToTensor()
        self.default_label = default_label
        self.samples = []

        if os.path.isdir(root_dir):
            for name in sorted(os.listdir(root_dir)):
                if name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append(os.path.join(root_dir, name))

        if len(self.samples) == 0:
            raise FileNotFoundError(f"No backdoored MNIST images found in {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        image = Image.open(path).convert('L')
        x = self.transform(image)
        label = self.default_label

        base = os.path.basename(path)
        if '_label_' in base:
            label_part = base.rsplit('_label_', 1)[1].split('.', 1)[0]
            try:
                label = int(label_part)
            except ValueError:
                label = self.default_label

        return x, int(label)


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
        self.fc3 = nn.Linear(28 * 28, 28 * 28)  # output

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

            # logits_z, logits_y, x_hat, mu, logvar = model(x, mode='with_reconstruction')
            logits_z, logits_y, mu, logvar = model(x, mode='distribution')
            # VAE two loss: KLD + MSE
            H_p_q = loss_fn(logits_y, y)

            KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar).cuda()
            KLD = torch.sum(KLD_element).mul_(-0.5).cuda()
            KLD_mean = torch.mean(KLD_element).mul_(-0.5).cuda()

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



def split_client_indices(client_subset: Subset, erase_ratio: float, seed: int):
    """
    Split a client's Subset into erase and remain subsets, returning:
      erase_subset, remain_subset, erase_global_indices, remain_global_indices
    where *_global_indices index into the ORIGINAL train_set.
    """
    rng = np.random.RandomState(seed)

    global_indices = np.array(client_subset.indices)  # indices into train_set
    n = len(global_indices)
    n_erase = max(1, int(round(n * erase_ratio)))

    perm = rng.permutation(n)
    erase_pos = perm[:n_erase]
    remain_pos = perm[n_erase:]

    erase_global = global_indices[erase_pos].tolist()
    remain_global = global_indices[remain_pos].tolist()

    erase_subset = Subset(client_subset.dataset, erase_global)
    remain_subset = Subset(client_subset.dataset, remain_global)

    return erase_subset, remain_subset, erase_global, remain_global


def make_loader(ds, batch_size, shuffle=True, num_workers=1, drop_last=False):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)


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


def prepare_unl(erasing_dataset, dataloader_remaining_after_aux, model, loss_fn, args, noise_flag):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    step = 0
    backdoor_acc_list = []

    print(len(erasing_dataset.dataset))
    for epoch in range(args.local_epochs):
        model.train()
        for (x_e, y_e), (x_r, y_r) in zip(erasing_dataset, dataloader_remaining_after_aux):
            x_e, y_e = x_e.to(args.device), y_e.to(args.device)  # (B, C, H, W), (B, 10)
            x_r, y_r = x_r.to(args.device), y_r.to(args.device)  # (B, C, H, W), (B, 10)

            logits_z_e, logits_y_e, x_hat_e, mu_e, logvar_e = model(x_e, mode='with_reconstruction')  # (B, C* h* w), (B, N, 10)

            logits_z_r, logits_y_r, x_hat_r, mu_r, logvar_r = model(x_r, mode='with_reconstruction')

            KLD_element = mu_e.pow(2).add_(logvar_e.exp()).mul_(-1).add_(1).add_(logvar_e).cuda()
            KLD_mean = torch.mean(KLD_element).mul_(-0.5).cuda()
            H_p_q = loss_fn(logits_y_e, y_e)
            # loss = args.beta * KLD_mean - args.unlearn_learning_rate * H_p_q

            H_p_q2 = loss_fn(logits_y_r, y_r)
            KLD_element2 = mu_r.pow(2).add_(logvar_r.exp()).mul_(-1).add_(1).add_(logvar_r).to(args.device)
            KLD_mean2 = torch.mean(KLD_element2).mul_(-0.5).to(args.device)

            loss = 0.5*(args.beta * KLD_mean - H_p_q) + 0.5*(args.beta * KLD_mean2 + H_p_q2)

            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 5, norm_type=2.0, error_if_nonfinite=False)
            optimizer.step()
            acc_back = (logits_y_e.argmax(dim=1) == y_e).float().mean().item()
            # backdoor_acc_list.append(acc_r)
            metrics = {
                'acc_back': acc_back,
                'loss1': loss.item(),
                'KLD_mean': KLD_mean.item(),
                # '1-JS(p,q)': JS_p_q,
                'mu': torch.mean(mu_e).item(),
                # 'KLD': KLD.item(),
                # 'KLD_mean': KLD_mean.item(),
            }
            # if epoch == args.num_epochs - 1:
            #     mu_list.append(torch.mean(mu).item())
            #     sigma_list.append(sigma)
            if step % len(erasing_dataset) % 10000 == 0:
                print(f'[{epoch}/{0 + args.num_epochs}:{step % len(erasing_dataset):3d}] '
                      + ', '.join([f'{k} {v:.3f}' for k, v in metrics.items()]))
                x_cpu = x_e.cpu().data
                x_cpu = x_cpu.clamp(0, 1)
                x_cpu = x_cpu.view(x_cpu.size(0), 1, 28, 28)
                grid = torchvision.utils.make_grid(x_cpu, nrow=4)
                # plt.imshow(np.transpose(grid, (1, 2, 0)))  # 交换维度，从GBR换成RGB
                # plt.show()

                print(acc_back)
                print("print x grad")
                # print(updated_x)
            backdoor_acc = eva_vib(model, erasing_dataset, args, name='on erased data', epoch=999)
            model.train()
            if backdoor_acc < 0.1:
                break
    print("backdoor_acc_list", backdoor_acc_list)
    return model


def federated_unlearning_one_round(
    global_model: nn.Module,
    train_set,
    client_subsets,
    args,
    num_unlearn_clients=3,
    erase_ratio=0.05,
    base_seed=0,
    loss_fn=None,
    backdoor_dataset=None,
    backdoor_client_id=None,
    round_idx=1,
):
    """
    One federated unlearning round:
      - pick num_unlearn_clients clients
      - each does prepare_unl on 5% erase + 95% remain
      - in round 1, only the backdoor unlearning client uploads
      - in later rounds, non-chosen clients continue normal local training
      - FedAvg aggregate uploaded client models

    Returns:
      new_global_model, chosen_client_ids, client_data_info
    where client_data_info[cid] stores indices and loaders for chosen clients.
    """
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    reconstruction_function = nn.MSELoss(reduction='mean')

    rng = np.random.RandomState(base_seed)
    all_client_ids = list(range(len(client_subsets)))
    chosen_client_ids = rng.choice(all_client_ids, size=num_unlearn_clients, replace=False).tolist()
    if backdoor_dataset is not None and backdoor_client_id is None:
        backdoor_client_id = chosen_client_ids[0]
    if round_idx == 1 and backdoor_client_id is not None:
        chosen_client_ids = [backdoor_client_id]
    print("Chosen clients for unlearning:", chosen_client_ids)
    if backdoor_dataset is not None:
        print(f"Inject backdoored data into unlearning client: {backdoor_client_id}")

    client_weights = []
    client_data_info = {}  # store erase/remain indices and loaders for chosen clients

    # cache global weights once
    global_sd = copy.deepcopy(global_model.state_dict())


    for cid in all_client_ids:

        # --- local unlearning starting from current global model ---
        local_model = copy.deepcopy(global_model)
        local_model.load_state_dict(global_sd)
        local_model.to(args.device)


        if cid not in chosen_client_ids:
            if round_idx == 1:
                del local_model
                torch.cuda.empty_cache()
                continue

            local_model.train()
            optimizer_local = torch.optim.Adam(local_model.parameters(), lr=args.lr)
            client_loader = make_loader(
                client_subsets[cid],
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=1,
            )

            for _ in range(args.local_epochs):
                local_model = vib_train_original_IB(
                    client_loader,
                    local_model,
                    optimizer_local,
                    loss_fn,
                    reconstruction_function,
                    args,
                )

            client_weights.append(copy.deepcopy(local_model.state_dict()))
        else:
            # --- build erase/remain subsets + loaders for this client ---
            using_backdoor_erase = backdoor_dataset is not None and cid == backdoor_client_id
            if using_backdoor_erase:
                erase_subset = backdoor_dataset
                remain_subset = client_subsets[cid]
                erase_idx = [f"backdoor:{i}" for i in range(len(backdoor_dataset))]
                remain_idx = list(getattr(client_subsets[cid], "indices", range(len(client_subsets[cid]))))
            else:
                erase_subset, remain_subset, erase_idx, remain_idx = split_client_indices(
                    client_subsets[cid],
                    erase_ratio=erase_ratio,
                    seed=base_seed + 1000 + cid,  # make each client split deterministic but different
                )

            erase_loader = make_loader(erase_subset, batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=not using_backdoor_erase)
            remain_loader = make_loader(remain_subset, batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=True)

            client_data_info[cid] = {
                "erase_indices": erase_idx,         # indices into train_set
                "remain_indices": remain_idx,       # indices into train_set
                "erase_loader": erase_loader,
                "remain_loader": remain_loader,
                "erase_size": len(erase_subset),
                "remain_size": len(remain_subset),
            }

            local_model = prepare_unl(
                erasing_dataset=erase_loader,
                dataloader_remaining_after_aux=remain_loader,
                model=local_model,
                loss_fn=loss_fn,
                args=args,
                noise_flag="no noise",
            )

            client_weights.append(copy.deepcopy(local_model.state_dict()))

        del local_model
        torch.cuda.empty_cache()

    # --- aggregate ---
    new_sd = FedAvg(client_weights)
    new_global_model = copy.deepcopy(global_model)
    new_global_model.load_state_dict(new_sd)
    new_global_model.to(args.device)
    new_global_model.eval()

    return new_global_model, chosen_client_ids, client_data_info


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
args.dataset = 'MNIST'
args.add_noise = False
args.beta = 0.0001
args.mse_rate = 0.1
args.lr = 0.0001
args.dimZ = 512
args.batch_size = 100
args.backdoor_dir = "backdoored_mnist"

args.num_clients = 10
args.global_rounds = 10       # number of global aggregation rounds
args.local_epochs = 10         # local epochs per client per round
args.frac = 1.0               # fraction of clients per round (1.0 = all clients)





print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))

device = args.device
print("device", device)

backdoor_dataset = None
backdoor_loader = None

if args.dataset == 'MNIST':
    transform = T.Compose([
        T.ToTensor()

    ])
    trans_mnist = transforms.Compose([transforms.ToTensor(), ])
    train_set = MNIST('/home/wwq/Data/data/mnist', train=True, transform=trans_mnist, download=True)
    test_set = MNIST('/home/wwq/Data/data/mnist', train=False, transform=trans_mnist, download=False)
    train_set_no_aug = train_set
    backdoor_dir = args.backdoor_dir
    if not os.path.isabs(backdoor_dir):
        backdoor_dir = os.path.join(os.path.dirname(__file__), backdoor_dir)
    backdoor_dataset = BackdooredMNISTFolder(backdoor_dir, transform=trans_mnist, default_label=7)
    backdoor_loader = DataLoader(backdoor_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)
    print(f"Loaded {len(backdoor_dataset)} backdoored MNIST samples from {backdoor_dir}")
elif args.dataset == 'CIFAR10':
    train_transform = T.Compose([
        T.ToTensor(),
    ])
    test_transform = T.Compose([T.ToTensor(),
                                ])
    train_set = CIFAR10('/home/wwq/Data/data/cifar', train=True, transform=train_transform, download=True)
    test_set = CIFAR10('/home/wwq/Data/data/cifar', train=False, transform=test_transform, download=False)
    train_set_no_aug = CIFAR10('/home/wwq/Data/data/cifar', train=True, transform=test_transform, download=False)




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




#load model directly

ckpt_path = "global_vib_mnist_fedavg_backdoored.pt"
ckpt = torch.load(ckpt_path, map_location=args.device)

global_vib, _ = init_vib(args)          # rebuild same architecture
global_vib.load_state_dict(ckpt["model_state"])
global_vib.to(args.device)
global_vib.eval()

print("Loaded checkpoint from round:", ckpt.get("round", "unknown"))

_ = eva_vib(global_vib, test_loader, args, name="loaded model", epoch=ckpt.get("round", 0))
if backdoor_loader is not None:
    _ = eva_vib(global_vib, backdoor_loader, args, name="loaded model backdoor attack", epoch=ckpt.get("round", 0))




# record time for unlearning
loss_fn = nn.CrossEntropyLoss()

start_time = time.time()

# run federated unlearning rounds
for rnd in range(1, args.global_rounds + 1):
    new_global_vib, chosen_ids, client_data_info = federated_unlearning_one_round(
        global_model=global_vib,
        train_set=train_set,
        client_subsets=client_subsets,
        args=args,
        num_unlearn_clients=3,
        erase_ratio=0.05,
        base_seed=seed,
        loss_fn=loss_fn,
        backdoor_dataset=backdoor_dataset,
        round_idx=rnd,
    )
    global_vib = new_global_vib
    _ = eva_vib(new_global_vib, test_loader, args, name=f"after federated unlearning round {rnd}", epoch=rnd)
    if backdoor_loader is not None:
        _ = eva_vib(new_global_vib, backdoor_loader, args, name=f"after federated unlearning round {rnd} backdoor attack", epoch=rnd)

print("Unlearning completed.")

end_time = time.time()
running_time = end_time - start_time
print(f'unlearning with dp {running_time} seconds')


# evaluate
_ = eva_vib(new_global_vib, test_loader, args, name="after federated unlearning", epoch=0)
if backdoor_loader is not None:
    _ = eva_vib(new_global_vib, backdoor_loader, args, name="after federated unlearning backdoor attack", epoch=0)


# inspect chosen clients' erase/remain splits
for cid in chosen_ids:
    info = client_data_info[cid]
    print(f"\nClient {cid}: erase={info['erase_size']} remain={info['remain_size']}")
    print("First 10 erase indices:", info["erase_indices"][:10])
    print("First 10 remain indices:", info["remain_indices"][:10])
