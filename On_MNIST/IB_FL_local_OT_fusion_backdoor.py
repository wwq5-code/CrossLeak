import sys

sys.argv = ['']
del sys

import os
os.environ["TQDM_DISABLE"] = "1"     # 放在导入 tqdm 前


import csv
import math
import itertools
import argparse
import json
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
    parser.add_argument('--ot_fusion_alpha', type=float, default=1.0)
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


def _linear_ot_permutation(ref_weight, src_weight):
    """
    Match source hidden neurons to reference hidden neurons by cosine cost.
    This is a discrete OT/Hungarian matching with uniform neuron mass.
    """
    ref_feat = ref_weight.detach().reshape(ref_weight.shape[0], -1).float().cpu()
    src_feat = src_weight.detach().reshape(src_weight.shape[0], -1).float().cpu()

    ref_feat = F.normalize(ref_feat, dim=1)
    src_feat = F.normalize(src_feat, dim=1)
    cost = 1.0 - torch.mm(src_feat, ref_feat.t()).numpy()

    try:
        from scipy.optimize import linear_sum_assignment
        src_ind, ref_ind = linear_sum_assignment(cost)
        perm = np.empty_like(ref_ind)
        perm[ref_ind] = src_ind
        return torch.as_tensor(perm, dtype=torch.long, device=src_weight.device)
    except Exception:
        # Greedy fallback for environments without scipy.
        cost_t = torch.as_tensor(cost)
        used_src = set()
        perm = []
        for ref_idx in range(cost_t.shape[1]):
            order = torch.argsort(cost_t[:, ref_idx]).tolist()
            src_idx = next(i for i in order if i not in used_src)
            used_src.add(src_idx)
            perm.append(src_idx)
        return torch.as_tensor(perm, dtype=torch.long, device=src_weight.device)


def _align_linear_hidden_layer(aligned_sd, ref_sd, layer_prefix, next_weight_name):
    """
    Align a hidden Linear layer and the following layer's input columns.
    """
    weight_name = f"{layer_prefix}.weight"
    bias_name = f"{layer_prefix}.bias"

    if weight_name not in aligned_sd or next_weight_name not in aligned_sd:
        return
    if aligned_sd[weight_name].ndim != 2 or aligned_sd[next_weight_name].ndim != 2:
        return

    perm = _linear_ot_permutation(ref_sd[weight_name], aligned_sd[weight_name])
    aligned_sd[weight_name] = aligned_sd[weight_name][perm].clone()
    if bias_name in aligned_sd:
        aligned_sd[bias_name] = aligned_sd[bias_name][perm].clone()
    aligned_sd[next_weight_name] = aligned_sd[next_weight_name][:, perm].clone()


def ot_align_to_global(unlearned_sd, global_sd):
    """
    OT-align permutation-symmetric hidden layers before model fusion.

    For this VIB architecture the safe permutation symmetries are the hidden
    units inside the MLP blocks. We keep latent dimensions and image/output
    coordinates fixed because they have shared semantic meaning across encoder,
    approximator, decoder, and reconstruction head.
    """
    aligned_sd = copy.deepcopy(unlearned_sd)
    _align_linear_hidden_layer(aligned_sd, global_sd, "encoder.fc1", "encoder.fc_out.weight")
    _align_linear_hidden_layer(aligned_sd, global_sd, "approximator.fc1", "approximator.fc2.weight")
    _align_linear_hidden_layer(aligned_sd, global_sd, "decoder.fc1", "decoder.fc2.weight")
    return aligned_sd


def ot_fusion_unlearning(global_sd, unlearned_weights, fusion_alpha=1.0):
    """
    Fuse global and locally unlearned models after OT alignment.

    The fused point is on the linear mode-connectivity path between the
    original global model and the mean OT-aligned unlearned model:
        theta_fused = theta_global + alpha * mean(theta_unlearned_aligned - theta_global)
    alpha=1.0 gives the OT-aligned unlearned average; smaller values keep the
    result closer to the original global model.
    """
    if len(unlearned_weights) == 0:
        return copy.deepcopy(global_sd)

    aligned_weights = [ot_align_to_global(w, global_sd) for w in unlearned_weights]
    fused_sd = copy.deepcopy(global_sd)

    for k, global_v in fused_sd.items():
        if torch.is_floating_point(global_v):
            avg_delta = torch.zeros_like(global_v)
            for aligned_sd in aligned_weights:
                avg_delta += aligned_sd[k].to(global_v.device) - global_v
            avg_delta = avg_delta / len(aligned_weights)
            fused_sd[k] = global_v + fusion_alpha * avg_delta
        else:
            fused_sd[k] = global_v

    return fused_sd


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


def fine_tune_on_remaining(remaining_loader, model, loss_fn, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()

    for epoch in range(args.local_epochs):
        with tqdm(remaining_loader) as pbar:
            for step, (x, y) in enumerate(pbar):
                x, y = x.to(args.device), y.to(args.device)
                x.requires_grad = True

                logits_z, logits_y, mu, logvar = model(x, mode='distribution')
                H_p_q = loss_fn(logits_y, y)

                KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar).to(args.device)
                KLD_mean = torch.mean(KLD_element).mul_(-0.5).to(args.device)

                loss = args.beta * KLD_mean + H_p_q

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5, norm_type=2.0, error_if_nonfinite=False)
                optimizer.step()

                accuracy = (logits_y.argmax(dim=1) == y).float().mean()
                pbar.set_postfix(
                    loss=loss.item(),
                    accuracy=accuracy.item(),
                    I_xz=KLD_mean.item(),
                    lr=optimizer.param_groups[0]['lr'],
                )

    return model


def synchronize_device(args):
    if getattr(args, "device", None) is not None and args.device.type == "cuda":
        torch.cuda.synchronize(args.device)


def state_dict_nbytes(state_dict):
    total_bytes = 0
    for tensor in state_dict.values():
        if torch.is_tensor(tensor):
            total_bytes += tensor.numel() * tensor.element_size()
    return total_bytes


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
):
    """
    One OTFU federated unlearning round:
      - pick num_unlearn_clients clients
      - each chosen client does prepare_unl on 5% erase + 95% remain
      - non-chosen clients are skipped
      - OT-align locally unlearned models to the original global model
      - fuse original global and OT-aligned unlearned models

    Returns:
      new_global_model, chosen_client_ids, client_data_info, round_parallel_time, communication_cost_bytes
    where client_data_info[cid] stores indices and loaders for chosen clients.
    round_parallel_time is max(client local compute time) because clients run in parallel.
    communication_cost_bytes is total download + upload bytes for all chosen clients.
    """
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    rng = np.random.RandomState(base_seed)
    all_client_ids = list(range(len(client_subsets)))
    chosen_client_ids = rng.choice(all_client_ids, size=num_unlearn_clients, replace=False).tolist()
    if backdoor_dataset is not None and backdoor_client_id is None:
        backdoor_client_id = chosen_client_ids[0]
    print("Chosen clients for unlearning:", chosen_client_ids)
    if backdoor_dataset is not None:
        print(f"Inject backdoored data into unlearning client: {backdoor_client_id}")

    unlearned_weights = []
    client_data_info = {}  # store erase/remain indices and loaders for chosen clients
    client_times = []

    # cache global weights once
    global_sd = copy.deepcopy(global_model.state_dict())
    model_size_bytes = state_dict_nbytes(global_sd)
    fusion_alpha = getattr(args, "ot_fusion_alpha", 1.0)


    for cid in all_client_ids:

        if cid not in chosen_client_ids:
            continue

        # --- local unlearning starting from current global model ---
        local_model = copy.deepcopy(global_model)
        local_model.load_state_dict(global_sd)
        local_model.to(args.device)

        synchronize_device(args)
        client_start_time = time.time()

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
        local_model = fine_tune_on_remaining(
            remaining_loader=remain_loader,
            model=local_model,
            loss_fn=loss_fn,
            args=args,
        )

        synchronize_device(args)
        client_times.append(time.time() - client_start_time)
        unlearned_weights.append(copy.deepcopy(local_model.state_dict()))
        del local_model
        torch.cuda.empty_cache()

    # --- OTFU: OT-align unlearned models and fuse with the original global model ---
    new_sd = ot_fusion_unlearning(global_sd, unlearned_weights, fusion_alpha=fusion_alpha)
    new_global_model = copy.deepcopy(global_model)
    new_global_model.load_state_dict(new_sd)
    new_global_model.to(args.device)
    new_global_model.eval()
    round_parallel_time = max(client_times) if client_times else 0.0
    communication_cost_bytes = 2 * len(chosen_client_ids) * model_size_bytes

    return new_global_model, chosen_client_ids, client_data_info, round_parallel_time, communication_cost_bytes


def build_erased_data_loader(train_set, backdoor_dataset, client_data_info, args):
    normal_erase_indices = []
    include_backdoor = False
    for info in client_data_info.values():
        for idx in info["erase_indices"]:
            if isinstance(idx, str) and idx.startswith("backdoor:"):
                include_backdoor = True
            else:
                normal_erase_indices.append(idx)

    datasets_to_eval = []
    if normal_erase_indices:
        datasets_to_eval.append(Subset(train_set, normal_erase_indices))
    if include_backdoor and backdoor_dataset is not None:
        datasets_to_eval.append(backdoor_dataset)
    if not datasets_to_eval:
        return None

    erased_dataset = datasets_to_eval[0] if len(datasets_to_eval) == 1 else ConcatDataset(datasets_to_eval)
    return make_loader(erased_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)


def configure_mnist_backdoor_otfu_args():
    args = args_parser()
    args.gpu = 0
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
    args.num_unlearn_clients = 3
    args.unlearning_ratio = 0.05
    args.global_rounds = 1
    args.local_epochs = 1
    args.frac = 1.0
    args.seeds = [0, 1, 2, 3, 4]
    args.results_dir = os.path.dirname(os.path.abspath(__file__))
    return args


def load_mnist_backdoor_data(args, seed):
    trans_mnist = transforms.Compose([transforms.ToTensor()])
    train_set = MNIST('/home/wwq/Data/data/mnist', train=True, transform=trans_mnist, download=True)
    test_set = MNIST('/home/wwq/Data/data/mnist', train=False, transform=trans_mnist, download=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=1)
    client_subsets = split_dataset_iid(train_set, args.num_clients, seed=seed)
    args.x_shape = (1, 28, 28)

    backdoor_dir = args.backdoor_dir
    if not os.path.isabs(backdoor_dir):
        backdoor_dir = os.path.join(os.path.dirname(__file__), backdoor_dir)
    backdoor_dataset = BackdooredMNISTFolder(backdoor_dir, transform=trans_mnist, default_label=7)
    backdoor_loader = DataLoader(backdoor_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)
    print(f"Loaded {len(backdoor_dataset)} backdoored MNIST samples from {backdoor_dir}")

    return train_set, test_set, train_loader, test_loader, client_subsets, backdoor_dataset, backdoor_loader


def load_global_mnist_backdoor_model(args):
    ckpt_path = "global_vib_mnist_fedavg_backdoored.pt"
    ckpt = torch.load(ckpt_path, map_location=args.device)
    global_vib, _ = init_vib(args)
    global_vib.load_state_dict(ckpt["model_state"])
    global_vib.to(args.device)
    global_vib.eval()
    print("Loaded checkpoint from round:", ckpt.get("round", "unknown"))
    return global_vib, ckpt


def set_random_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def flatten_result_metrics(result):
    metrics = {
        "acc_after": result["acc_after"],
        "backdoor_acc_before": result["backdoor_acc_before"],
        "backdoor_acc_after": result["backdoor_acc_after"],
        "erased_acc": result["erased_acc"],
        "running_time": result["running_time"],
        "communication_cost_bytes": result["communication_cost_bytes"],
        "communication_cost_mb": result["communication_cost_mb"],
    }
    return metrics


def mean_std(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return float(np.mean(arr)), std


def summarize_seed_sweep(all_seed_results):
    grouped = {}
    for seed_result in all_seed_results:
        seed = seed_result["seed"]
        for result in seed_result["results"]:
            num_clients = result["num_unlearn_clients"]
            grouped.setdefault(num_clients, []).append((seed, flatten_result_metrics(result)))

    summary = []
    for num_clients in sorted(grouped):
        metric_names = sorted({name for _, metrics in grouped[num_clients] for name in metrics})
        row = {
            "num_unlearn_clients": num_clients,
            "num_seeds": len(grouped[num_clients]),
            "seeds": [seed for seed, _ in grouped[num_clients]],
        }
        for metric_name in metric_names:
            values = [metrics.get(metric_name, float("nan")) for _, metrics in grouped[num_clients]]
            metric_mean, metric_std = mean_std(values)
            row[f"{metric_name}_mean"] = metric_mean
            row[f"{metric_name}_std"] = metric_std
        summary.append(row)
    return summary


def print_seed_sweep_summary(summary):
    print("\n===== MNIST Backdoor OTFU multi-seed mean/std summary =====")
    for row in summary:
        parts = [
            f"clients={row['num_unlearn_clients']}",
            f"num_seeds={row['num_seeds']}",
        ]
        metric_bases = sorted(
            key[:-5] for key in row
            if key.endswith("_mean") and f"{key[:-5]}_std" in row
        )
        for metric in metric_bases:
            parts.append(f"{metric}={row[f'{metric}_mean']:.4f}±{row[f'{metric}_std']:.4f}")
        print(", ".join(parts))


def save_seed_sweep_records(all_seed_results, summary, args):
    os.makedirs(args.results_dir, exist_ok=True)
    raw_path = os.path.join(args.results_dir, "otfu_backdoor_multi_seed_raw.json")
    summary_json_path = os.path.join(args.results_dir, "otfu_backdoor_multi_seed_summary.json")
    summary_csv_path = os.path.join(args.results_dir, "otfu_backdoor_multi_seed_summary.csv")

    with open(raw_path, "w") as f:
        json.dump(all_seed_results, f, indent=2)
    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=2)

    fieldnames = sorted({key for row in summary for key in row})
    preferred = ["num_unlearn_clients", "num_seeds", "seeds"]
    fieldnames = preferred + [key for key in fieldnames if key not in preferred]
    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)

    print("\nSaved multi-seed records:")
    print(f"  raw: {raw_path}")
    print(f"  summary_json: {summary_json_path}")
    print(f"  summary_csv: {summary_csv_path}")


def run_unlearning_clients_sweep_for_seed(args, seed):
    set_random_seed(seed)
    print(f"\n===== MNIST Backdoor OTFU seed={seed} =====")

    train_set, test_set, train_loader, test_loader, client_subsets, backdoor_dataset, backdoor_loader = load_mnist_backdoor_data(args, seed)
    global_vib, ckpt = load_global_mnist_backdoor_model(args)
    _ = eva_vib(global_vib, test_loader, args, name=f"loaded model (seed={seed})", epoch=ckpt.get("round", 0))
    backdoor_acc_before = eva_vib(
        global_vib,
        backdoor_loader,
        args,
        name=f"loaded model backdoor attack (seed={seed})",
        epoch=ckpt.get("round", 0),
    )

    loss_fn = nn.CrossEntropyLoss()
    sweep_results = []

    for num_unlearn_clients in range(1, 6):
        print(f"\n===== MNIST Backdoor OTFU test: num_unlearn_clients={num_unlearn_clients} =====")
        args.num_unlearn_clients = num_unlearn_clients

        run_model = copy.deepcopy(global_vib)
        run_model.to(args.device)
        run_model.eval()

        running_time = 0.0
        communication_cost_bytes = 0
        client_data_info = {}
        chosen_ids = []
        new_global_vib = run_model
        for rnd in range(1, args.global_rounds + 1):
            new_global_vib, chosen_ids, client_data_info, round_parallel_time, round_communication_cost_bytes = federated_unlearning_one_round(
                global_model=new_global_vib,
                train_set=train_set,
                client_subsets=client_subsets,
                args=args,
                num_unlearn_clients=args.num_unlearn_clients,
                erase_ratio=args.unlearning_ratio,
                base_seed=seed,
                loss_fn=loss_fn,
                backdoor_dataset=backdoor_dataset,
            )
            running_time += round_parallel_time
            communication_cost_bytes += round_communication_cost_bytes
            print(
                f"Round {rnd} parallel client time: {round_parallel_time:.4f}s, "
                f"communication_cost={round_communication_cost_bytes / (1024 ** 2):.4f} MB"
            )

        acc_after = eva_vib(
            new_global_vib,
            test_loader,
            args,
            name=f"after OT fusion backdoor unlearning ({num_unlearn_clients} clients)",
            epoch=args.global_rounds,
        )
        backdoor_acc_after = eva_vib(
            new_global_vib,
            backdoor_loader,
            args,
            name=f"after OT fusion backdoor attack ({num_unlearn_clients} clients)",
            epoch=args.global_rounds,
        )

        erased_loader = build_erased_data_loader(train_set, backdoor_dataset, client_data_info, args)
        erased_acc = float("nan")
        if erased_loader is not None:
            erased_acc = eva_vib(
                new_global_vib,
                erased_loader,
                args,
                name=f"erased data ({num_unlearn_clients} clients)",
                epoch=args.global_rounds,
            )

        result = {
            "num_unlearn_clients": num_unlearn_clients,
            "seed": seed,
            "chosen_ids": chosen_ids,
            "acc_after": acc_after,
            "backdoor_acc_before": backdoor_acc_before,
            "backdoor_acc_after": backdoor_acc_after,
            "erased_acc": erased_acc,
            "running_time": running_time,
            "communication_cost_bytes": communication_cost_bytes,
            "communication_cost_mb": communication_cost_bytes / (1024 ** 2),
        }
        sweep_results.append(result)

        print(
            "MNIST Backdoor OTFU sweep result: "
            f"num_unlearn_clients={num_unlearn_clients}, "
            f"chosen_ids={chosen_ids}, "
            f"acc_after={acc_after:.4f}, "
            f"backdoor_acc_before={backdoor_acc_before:.4f}, "
            f"backdoor_acc_after={backdoor_acc_after:.4f}, "
            f"erased_acc={erased_acc:.4f}, "
            f"running_time={running_time:.4f}s, "
            f"communication_cost={communication_cost_bytes / (1024 ** 2):.4f}MB"
        )

        del run_model, new_global_vib
        torch.cuda.empty_cache()

    print(f"\n===== MNIST Backdoor OTFU sweep summary (seed={seed}) =====")
    for result in sweep_results:
        print(
            f"clients={result['num_unlearn_clients']}, "
            f"seed={result['seed']}, "
            f"acc_after={result['acc_after']:.4f}, "
            f"backdoor_acc_before={result['backdoor_acc_before']:.4f}, "
            f"backdoor_acc_after={result['backdoor_acc_after']:.4f}, "
            f"erased_acc={result['erased_acc']:.4f}, "
            f"running_time={result['running_time']:.4f}s, "
            f"communication_cost={result['communication_cost_mb']:.4f}MB"
        )

    return sweep_results


def test_unlearning_clients_sweep():
    args = configure_mnist_backdoor_otfu_args()
    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))
    print("device", args.device)
    print("multi-seed MNIST Backdoor OTFU evaluation seeds:", args.seeds)

    all_seed_results = []
    for seed in args.seeds:
        seed_results = run_unlearning_clients_sweep_for_seed(args, seed)
        all_seed_results.append({
            "seed": seed,
            "results": seed_results,
        })

    summary = summarize_seed_sweep(all_seed_results)
    print_seed_sweep_summary(summary)
    save_seed_sweep_records(all_seed_results, summary, args)
    return all_seed_results, summary


if __name__ == "__main__":
    test_unlearning_clients_sweep()
