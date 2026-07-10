import sys

sys.argv = ['']
del sys

import csv
import os
os.environ["TQDM_DISABLE"] = "1"     # 放在导入 tqdm 前


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

from torch.nn.functional import cosine_similarity
from tqdm import tqdm



def args_parser():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset', choices=['MNIST', 'CIFAR10'], default='CIFAR10')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs for VIBI.')
    parser.add_argument('--explainer_type', choices=['Unet', 'ResNet_2x', 'ResNet_4x', 'ResNet_8x'],
                        default='ResNet_4x')
    parser.add_argument('--xpl_channels', type=int, choices=[1, 3], default=1)
    parser.add_argument('--k', type=int, default=12, help='Number of chunks.')
    parser.add_argument('--beta', type=float, default=0, help='beta in objective J = I(y,t) - beta * I(x,t).')
    parser.add_argument('--unlearning_ratio', type=float, default=0.1)
    parser.add_argument('--mia_max_samples', type=int, default=1000,
                        help='Maximum member/non-member samples used for membership inference attack.')
    parser.add_argument('--mia_attack_epochs', type=int, default=200,
                        help='Training epochs for learned membership inference attack.')
    parser.add_argument('--mia_attack_lr', type=float, default=0.01,
                        help='Learning rate for learned membership inference attack.')
    parser.add_argument('--num_samples', type=int, default=4,
                        help='Number of samples used for estimating expectation over p(t|x).')
    args = parser.parse_args()
    return args


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
    def __init__(self, encoder, approximator, decoder, reconstruction_dim):
        super().__init__()

        self.encoder = encoder
        self.approximator = approximator
        self.decoder = decoder
        self.fc3 = nn.Linear(reconstruction_dim, reconstruction_dim)

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
        reconstruction_dim = 28 * 28
        decoder = LinearModel(n_feature=args.dimZ, n_output=reconstruction_dim)
        encoder = CNNEncoder(in_channels=1, dim_z=args.dimZ * 2)

        lr = args.lr

    elif args.dataset == 'CIFAR10':

        approximator = LinearModel(n_feature=args.dimZ)

        encoder = resnet18(3, args.dimZ * 2)  # resnet18(1, 49*2)
        reconstruction_dim = 3 * 32 * 32
        decoder = LinearModel(n_feature=args.dimZ, n_output=reconstruction_dim)
        lr = args.lr

    elif args.dataset == 'CIFAR100':
        approximator = LinearModel(n_feature=args.dimZ, n_output=100)
        encoder = resnet18(3, args.dimZ * 2)  # resnet18(1, 49*2)
        reconstruction_dim = 3 * 32 * 32
        decoder = LinearModel(n_feature=args.dimZ, n_output=reconstruction_dim)
        lr = args.lr

    vib = VIB(encoder, approximator, decoder, reconstruction_dim)
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

            KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar).to(args.device)
            KLD = torch.sum(KLD_element).mul_(-0.5).to(args.device)
            KLD_mean = torch.mean(KLD_element).mul_(-0.5).to(args.device)

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
def eva_vib(vib, dataloader_erase, args, name='test', epoch=999, verbose=True):
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
    if verbose:
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



def _dataset_label_at(dataset, idx):
    if hasattr(dataset, "targets"):
        return int(dataset.targets[idx])
    if hasattr(dataset, "labels"):
        return int(dataset.labels[idx])
    _, y = dataset[idx]
    return int(y.argmax().item()) if torch.is_tensor(y) and y.ndim > 0 else int(y)


def _normalize_target_classes(target_classes):
    if target_classes is None:
        return None
    if isinstance(target_classes, (int, np.integer)):
        return [int(target_classes)]
    return [int(label) for label in target_classes]


def split_client_indices(client_subset: Subset, erase_ratio: float, seed: int, target_classes=None):
    """
    Split a client's Subset into erase and remain subsets, returning:
      erase_subset, remain_subset, erase_global_indices, remain_global_indices
    where *_global_indices index into the ORIGINAL train_set.
    If target_classes is provided, erase samples are preferentially drawn from
    those classes so the erased 5% carries a configurable label-level signal.
    """
    rng = np.random.RandomState(seed)

    global_indices = np.array(client_subset.indices)  # indices into train_set
    n = len(global_indices)
    n_erase = max(1, int(round(n * erase_ratio)))
    target_classes = _normalize_target_classes(target_classes)

    if target_classes is None or len(target_classes) == 0:
        perm = rng.permutation(n)
        erase_pos = perm[:n_erase]
        remain_pos = perm[n_erase:]
    else:
        labels = np.array([_dataset_label_at(client_subset.dataset, int(idx)) for idx in global_indices])
        target_pos_by_class = []
        for target_class in target_classes:
            pos = np.where(labels == int(target_class))[0]
            rng.shuffle(pos)
            target_pos_by_class.append(pos)
        target_pos = np.concatenate(target_pos_by_class) if target_pos_by_class else np.array([], dtype=np.int64)
        other_pos = np.where(~np.isin(labels, np.array(target_classes)))[0]
        rng.shuffle(other_pos)

        selected = []
        per_class_quota = int(math.ceil(n_erase / len(target_classes)))
        for pos in target_pos_by_class:
            selected.extend(pos[:per_class_quota].tolist())
        if len(selected) >= n_erase:
            selected = rng.choice(np.array(selected), size=n_erase, replace=False).tolist()
        else:
            selected_set = set(selected)
            remaining_target = [int(pos) for pos in target_pos.tolist() if int(pos) not in selected_set]
            rng.shuffle(remaining_target)
            selected.extend(remaining_target[:max(0, n_erase - len(selected))])

        if len(selected) < n_erase:
            needed = n_erase - len(selected)
            selected.extend(other_pos[:needed].tolist())
            print(
                f"Warning: client has only {len(target_pos)} samples from target_classes={target_classes}; "
                f"filled {needed} erase samples from other classes."
            )
        erase_pos = np.array(selected[:n_erase], dtype=np.int64)
        erase_mask = np.zeros(n, dtype=bool)
        erase_mask[erase_pos] = True
        remain_pos = np.where(~erase_mask)[0]

    erase_global = global_indices[erase_pos].tolist()
    remain_global = global_indices[remain_pos].tolist()

    erase_subset = Subset(client_subset.dataset, erase_global)
    remain_subset = Subset(client_subset.dataset, remain_global)

    return erase_subset, remain_subset, erase_global, remain_global


def make_loader(ds, batch_size, shuffle=True, num_workers=1, drop_last=False):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)


def resolve_unlearning_target_classes(args):
    if not getattr(args, "class_conditional_unlearning", False):
        return None
    class_range = int(getattr(args, "unlearning_class_range", 1))
    if class_range <= 0:
        return None
    candidates = getattr(args, "unlearn_target_classes", None)
    if candidates is None:
        candidates = list(range(10))
    candidates = _normalize_target_classes(candidates)
    if len(candidates) == 0:
        return None
    if class_range > len(candidates):
        raise ValueError(
            f"unlearning_class_range={class_range} exceeds available target classes {candidates}"
        )
    return candidates[:class_range]


def resolve_existing_path(*candidates):
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return candidates[0]


def _rankdata_average(values):
    """Average ranks for ROC-AUC tie handling, implemented without sklearn/scipy."""
    values = np.asarray(values)
    order = np.argsort(values)
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)

    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j

    return ranks


def _binary_auc(member_scores, nonmember_scores):
    scores = np.concatenate([member_scores, nonmember_scores])
    labels = np.concatenate([
        np.ones(len(member_scores), dtype=np.int64),
        np.zeros(len(nonmember_scores), dtype=np.int64),
    ])
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = _rankdata_average(scores)
    sum_pos_ranks = ranks[labels == 1].sum()
    return float((sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _best_threshold_attack_acc(member_scores, nonmember_scores):
    thresholds = np.unique(np.concatenate([member_scores, nonmember_scores]))
    if len(thresholds) == 0:
        return float("nan"), float("nan")

    best_acc = 0.0
    best_threshold = thresholds[0]
    for threshold in thresholds:
        tpr = np.mean(member_scores >= threshold)
        tnr = np.mean(nonmember_scores < threshold)
        acc = 0.5 * (tpr + tnr)
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold

    return float(best_acc), float(best_threshold)


@torch.no_grad()
def _collect_mia_scores(vib, dataloader, args, max_samples=None):
    vib.eval()
    losses = []
    confidences = []
    entropies = []
    correctness = []
    collected = 0

    for x, y in dataloader:
        x, y = x.to(args.device), y.to(args.device)
        if y.ndim == 2:
            y = y.argmax(dim=1)

        _, logits_y, _, _, _ = vib(x, mode='with_reconstruction')
        probs = F.softmax(logits_y, dim=1)
        per_sample_loss = F.cross_entropy(logits_y, y, reduction='none')
        true_conf = probs.gather(1, y.view(-1, 1)).squeeze(1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)
        correct = (logits_y.argmax(dim=1) == y).float()

        losses.append(per_sample_loss.detach().cpu())
        confidences.append(true_conf.detach().cpu())
        entropies.append(entropy.detach().cpu())
        correctness.append(correct.detach().cpu())

        collected += x.size(0)
        if max_samples is not None and collected >= max_samples:
            break

    losses = torch.cat(losses)[:max_samples].numpy()
    confidences = torch.cat(confidences)[:max_samples].numpy()
    entropies = torch.cat(entropies)[:max_samples].numpy()
    correctness = torch.cat(correctness)[:max_samples].numpy()

    return {
        "loss": losses,
        "confidence": confidences,
        "entropy": entropies,
        "correctness": correctness,
    }


def build_mia_loaders(train_set, test_set, client_data_info, args, seed=0):
    erase_indices = []
    for info in client_data_info.values():
        erase_indices.extend(info["erase_indices"])

    if len(erase_indices) == 0:
        return None, None

    rng = np.random.RandomState(seed)
    max_samples = min(len(erase_indices), len(test_set), args.mia_max_samples)
    erase_indices = rng.choice(erase_indices, size=max_samples, replace=False).tolist()
    test_indices = rng.choice(np.arange(len(test_set)), size=max_samples, replace=False).tolist()

    member_subset = Subset(train_set, erase_indices)
    nonmember_subset = Subset(test_set, test_indices)

    member_loader = make_loader(member_subset, batch_size=args.batch_size, shuffle=False, num_workers=1)
    nonmember_loader = make_loader(nonmember_subset, batch_size=args.batch_size, shuffle=False, num_workers=1)

    return member_loader, nonmember_loader


def build_learned_mia_loaders(train_set, test_set, client_data_info, args, seed=0):
    erase_indices = []
    retain_indices = []
    for info in client_data_info.values():
        erase_indices.extend(info["erase_indices"])
        retain_indices.extend(info["remain_indices"])

    if len(erase_indices) == 0 or len(retain_indices) == 0:
        return None, None, None

    rng = np.random.RandomState(seed)
    n_attack = min(len(retain_indices), len(test_set), args.mia_max_samples)
    n_erase = min(len(erase_indices), args.mia_max_samples)

    retain_indices = rng.choice(retain_indices, size=n_attack, replace=False).tolist()
    nonmember_indices = rng.choice(np.arange(len(test_set)), size=n_attack, replace=False).tolist()
    erase_indices = rng.choice(erase_indices, size=n_erase, replace=False).tolist()

    retain_loader = make_loader(Subset(train_set, retain_indices), batch_size=args.batch_size, shuffle=False, num_workers=1)
    nonmember_loader = make_loader(Subset(test_set, nonmember_indices), batch_size=args.batch_size, shuffle=False, num_workers=1)
    erase_loader = make_loader(Subset(train_set, erase_indices), batch_size=args.batch_size, shuffle=False, num_workers=1)

    return erase_loader, retain_loader, nonmember_loader


def membership_inference_attack(vib, member_loader, nonmember_loader, args, name='model'):
    """
    Loss/confidence threshold MIA.
    Members are forgotten training samples; non-members are held-out test samples.
    Lower attack_acc/AUC after unlearning means weaker membership leakage.
    """
    member_metrics = _collect_mia_scores(vib, member_loader, args, max_samples=args.mia_max_samples)
    nonmember_metrics = _collect_mia_scores(vib, nonmember_loader, args, max_samples=args.mia_max_samples)

    attacks = {
        "loss": (-member_metrics["loss"], -nonmember_metrics["loss"]),
        "confidence": (member_metrics["confidence"], nonmember_metrics["confidence"]),
        "entropy": (-member_metrics["entropy"], -nonmember_metrics["entropy"]),
        "correctness": (member_metrics["correctness"], nonmember_metrics["correctness"]),
    }

    results = {}
    for attack_name, (member_scores, nonmember_scores) in attacks.items():
        auc = _binary_auc(member_scores, nonmember_scores)
        attack_acc, threshold = _best_threshold_attack_acc(member_scores, nonmember_scores)
        results[attack_name] = {
            "auc": auc,
            "attack_acc": attack_acc,
            "threshold": threshold,
            "advantage": 2.0 * max(auc - 0.5, 0.0) if not np.isnan(auc) else float("nan"),
        }

    print(f"\nMIA results for {name}:")
    print(
        f"  members={len(member_metrics['loss'])}, nonmembers={len(nonmember_metrics['loss'])}, "
        f"member_loss={member_metrics['loss'].mean():.4f}, "
        f"nonmember_loss={nonmember_metrics['loss'].mean():.4f}, "
        f"member_conf={member_metrics['confidence'].mean():.4f}, "
        f"nonmember_conf={nonmember_metrics['confidence'].mean():.4f}"
    )
    for attack_name, metrics in results.items():
        print(
            f"  {attack_name:11s} attack_acc={metrics['attack_acc']:.4f}, "
            f"auc={metrics['auc']:.4f}, advantage={metrics['advantage']:.4f}"
        )

    return results


@torch.no_grad()
def _collect_learned_mia_features(vib, dataloader, args, max_samples=None):
    vib.eval()
    features = []
    collected = 0

    for x, y in dataloader:
        x, y = x.to(args.device), y.to(args.device)
        if y.ndim == 2:
            y = y.argmax(dim=1)

        _, logits_y, _, _, _ = vib(x, mode='with_reconstruction')
        probs = F.softmax(logits_y, dim=1)
        sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
        true_conf = probs.gather(1, y.view(-1, 1)).squeeze(1)
        loss = F.cross_entropy(logits_y, y, reduction='none')
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)
        top1 = sorted_probs[:, 0]
        top2 = sorted_probs[:, 1]
        margin = top1 - top2
        correct = (logits_y.argmax(dim=1) == y).float()

        batch_features = torch.cat([
            sorted_probs,
            true_conf.unsqueeze(1),
            (-loss).unsqueeze(1),
            (-entropy).unsqueeze(1),
            margin.unsqueeze(1),
            correct.unsqueeze(1),
        ], dim=1)

        features.append(batch_features.detach().cpu())
        collected += x.size(0)
        if max_samples is not None and collected >= max_samples:
            break

    return torch.cat(features, dim=0)[:max_samples]


class LearnedMIAAttack(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.classifier(x).squeeze(1)


def _standardize_attack_features(train_features, *other_features):
    mean = train_features.mean(dim=0, keepdim=True)
    std = train_features.std(dim=0, keepdim=True).clamp_min(1e-6)
    scaled_train = (train_features - mean) / std
    scaled_others = [(features - mean) / std for features in other_features]
    return scaled_train, scaled_others


@torch.no_grad()
def _evaluate_attack_classifier(attack_model, features, labels):
    attack_model.eval()
    logits = attack_model(features)
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    acc = (preds == labels).float().mean().item()
    return probs.cpu().numpy(), acc


def learned_membership_inference_attack(vib, erase_loader, retain_loader, nonmember_loader, args, name='model'):
    """
    Train a neural MIA on retained members vs held-out non-members, then
    measure whether erased samples still look like members.
    """
    retain_features = _collect_learned_mia_features(vib, retain_loader, args, max_samples=args.mia_max_samples)
    nonmember_features = _collect_learned_mia_features(vib, nonmember_loader, args, max_samples=args.mia_max_samples)
    erase_features = _collect_learned_mia_features(vib, erase_loader, args, max_samples=args.mia_max_samples)

    n_attack = min(len(retain_features), len(nonmember_features))
    retain_features = retain_features[:n_attack]
    nonmember_features = nonmember_features[:n_attack]

    train_features = torch.cat([retain_features, nonmember_features], dim=0)
    train_labels = torch.cat([
        torch.ones(len(retain_features)),
        torch.zeros(len(nonmember_features)),
    ], dim=0)

    train_features, (erase_features,) = _standardize_attack_features(train_features, erase_features)
    train_labels = train_labels.float()

    attack_model = LearnedMIAAttack(train_features.size(1))
    optimizer = torch.optim.Adam(attack_model.parameters(), lr=args.mia_attack_lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    for _ in range(args.mia_attack_epochs):
        attack_model.train()
        logits = attack_model(train_features)
        loss = loss_fn(logits, train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_probs, train_acc = _evaluate_attack_classifier(attack_model, train_features, train_labels)
    erase_labels = torch.ones(len(erase_features))
    erase_probs, erase_member_acc = _evaluate_attack_classifier(attack_model, erase_features, erase_labels)

    retain_probs = train_probs[:len(retain_features)]
    nonmember_probs = train_probs[len(retain_features):]
    attack_auc = _binary_auc(retain_probs, nonmember_probs)

    result = {
        "attack_train_acc": train_acc,
        "attack_auc": attack_auc,
        "erase_member_prob": float(np.mean(erase_probs)),
        "erase_member_rate": float(np.mean(erase_probs >= 0.5)),
        "retain_member_prob": float(np.mean(retain_probs)),
        "nonmember_member_prob": float(np.mean(nonmember_probs)),
        "erase_as_member_acc": erase_member_acc,
        "num_erase": len(erase_features),
        "num_attack_members": len(retain_features),
        "num_attack_nonmembers": len(nonmember_features),
    }

    print(f"\nLearned MIA results for {name}:")
    print(
        f"  MIA acc={result['attack_train_acc']:.4f}, "
        f"MIA AUC={result['attack_auc']:.4f}"
    )

    return result


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return x, y, idx


class CrossCoder(nn.Module):
    """
    Lightweight CrossCoder for before/after model diffing.
    It learns one shared sparse latent index space and separate decoder
    directions for the original and unlearned global models.
    """
    def __init__(self, activation_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Linear(activation_dim, latent_dim)
        self.decoder_before = nn.Linear(latent_dim, activation_dim)
        self.decoder_after = nn.Linear(latent_dim, activation_dim)

    def encode_before(self, h):
        return F.relu(self.encoder(h))

    def encode_after(self, h):
        return F.relu(self.encoder(h))

    def forward(self, h_before, h_after):
        z_before = self.encode_before(h_before)
        z_after = self.encode_after(h_after)
        recon_before = self.decoder_before(z_before)
        recon_after = self.decoder_after(z_after)
        return z_before, z_after, recon_before, recon_after


def _unpack_batch(batch):
    if len(batch) == 3:
        x, y, idx = batch
    else:
        x, y = batch
        idx = None
    return x, y, idx


@torch.no_grad()
def _collect_crossleak_activations(before_model, after_model, dataloader, args, max_samples=None):
    before_model.eval()
    after_model.eval()
    h_before_list = []
    h_after_list = []
    label_list = []
    index_list = []
    logit_shift_list = []
    collected = 0

    for batch in dataloader:
        x, y, idx = _unpack_batch(batch)
        x = x.to(args.device)
        if y.ndim == 2:
            y = y.argmax(dim=1)

        before_params = before_model.encoder(x)
        after_params = after_model.encoder(x)
        dim_z = before_params.shape[1] // 2
        h_before = before_params[:, :dim_z]
        h_after = after_params[:, :dim_z]

        logits_before = before_model.approximator(h_before).reshape((x.size(0), 10))
        logits_after = after_model.approximator(h_after).reshape((x.size(0), 10))
        logit_shift = torch.norm(logits_before - logits_after, dim=1)

        h_before_list.append(h_before.detach().cpu())
        h_after_list.append(h_after.detach().cpu())
        label_list.append(y.detach().cpu())
        logit_shift_list.append(logit_shift.detach().cpu())
        if idx is None:
            idx = torch.arange(collected, collected + x.size(0))
        index_list.append(idx.detach().cpu())

        collected += x.size(0)
        if max_samples is not None and collected >= max_samples:
            break

    return {
        "before": torch.cat(h_before_list, dim=0)[:max_samples],
        "after": torch.cat(h_after_list, dim=0)[:max_samples],
        "labels": torch.cat(label_list, dim=0)[:max_samples],
        "indices": torch.cat(index_list, dim=0)[:max_samples],
        "logit_shift": torch.cat(logit_shift_list, dim=0)[:max_samples],
    }


def _standardize_pair_activations(h_before, h_after):
    stacked = torch.cat([h_before, h_after], dim=0)
    mean = stacked.mean(dim=0, keepdim=True)
    std = stacked.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (h_before - mean) / std, (h_after - mean) / std, mean, std


def train_crosscoder(h_before, h_after, args):
    h_before, h_after, mean, std = _standardize_pair_activations(h_before, h_after)
    latent_dim = getattr(args, "crossleak_latent_dim", 256)
    batch_size = getattr(args, "crossleak_batch_size", 256)
    epochs = getattr(args, "crossleak_epochs", 200)
    lr = getattr(args, "crossleak_lr", 1e-3)
    l1_lambda = getattr(args, "crossleak_l1_lambda", 1e-3)

    crosscoder = CrossCoder(h_before.size(1), latent_dim).to(args.device)
    optimizer = torch.optim.Adam(crosscoder.parameters(), lr=lr, weight_decay=1e-5)
    dataset = TensorDataset(h_before.float(), h_after.float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    for _ in range(epochs):
        crosscoder.train()
        for hb, ha in loader:
            hb = hb.to(args.device)
            ha = ha.to(args.device)
            zb, za, recon_b, recon_a = crosscoder(hb, ha)
            recon_loss = F.mse_loss(recon_b, hb) + F.mse_loss(recon_a, ha)
            sparse_loss = (zb.abs().mean() + za.abs().mean())
            loss = recon_loss + l1_lambda * sparse_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    crosscoder.eval()
    return crosscoder, mean, std


@torch.no_grad()
def _score_crossleak_samples(crosscoder, h_before, h_after, logit_shift, mean, std, args):
    hb = ((h_before - mean) / std).float().to(args.device)
    ha = ((h_after - mean) / std).float().to(args.device)
    z_before = crosscoder.encode_before(hb)
    z_after = crosscoder.encode_after(ha)

    dec_before = crosscoder.decoder_before.weight.detach().T
    dec_after = crosscoder.decoder_after.weight.detach().T
    norm_before = torch.norm(dec_before, dim=1)
    norm_after = torch.norm(dec_after, dim=1)
    rho = norm_before / (norm_before + norm_after + 1e-12)
    exclusivity = torch.abs(rho - 0.5) * 2.0
    tau = getattr(args, "crossleak_feature_tau", 0.15)
    feature_mask = exclusivity >= tau
    if int(feature_mask.sum().item()) == 0:
        top_k = min(getattr(args, "crossleak_top_features", 10), len(exclusivity))
        top_idx = torch.topk(exclusivity, k=top_k).indices
        feature_mask = torch.zeros_like(exclusivity, dtype=torch.bool)
        feature_mask[top_idx] = True

    latent_shift = torch.abs(z_before - z_after)
    weights = exclusivity * feature_mask.float()
    crosscoder_score = (latent_shift * weights.unsqueeze(0)).sum(dim=1)
    logit_shift = logit_shift.float().to(args.device)
    logit_shift = (logit_shift - logit_shift.mean()) / logit_shift.std().clamp_min(1e-6)
    crosscoder_score = (crosscoder_score - crosscoder_score.mean()) / crosscoder_score.std().clamp_min(1e-6)
    total_score = crosscoder_score + getattr(args, "crossleak_logit_weight", 0.1) * logit_shift

    return {
        "z_before": z_before.cpu(),
        "z_after": z_after.cpu(),
        "latent_shift": latent_shift.cpu(),
        "score": total_score.cpu(),
        "crosscoder_score": crosscoder_score.cpu(),
        "rho": rho.cpu(),
        "exclusivity": exclusivity.cpu(),
        "feature_mask": feature_mask.cpu(),
    }


def _label_histogram(labels, num_classes=10):
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes)
    total = counts.sum()
    if total == 0:
        return {}
    return {int(i): float(counts[i] / total) for i in range(num_classes) if counts[i] > 0}


def _top_class_items(class_scores, class_names, top_k=3):
    sorted_items = sorted(class_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
    return [
        {
            "label": int(label),
            "class_name": class_names[int(label)] if class_names else str(label),
            "score": float(score),
        }
        for label, score in sorted_items
    ]


def _feature_vote_label_scores(feature_info, class_names, args):
    """
    Infer deleted labels from deletion-sensitive CrossCoder features. This is
    better than class means when compensation features dominate sample-level
    shift scores.
    """
    before_bonus = getattr(args, "crossleak_before_exclusive_vote_bonus", 1.5)
    label_scores = {label: 0.0 for label in range(10)}

    for rank, feature in enumerate(feature_info):
        if feature.get("direction") != "before_exclusive_deleted":
            continue
        rank_weight = 1.0 / float(rank + 1)
        before_strength = max(float(feature.get("decoder_before_ratio", 0.5)) - 0.5, 0.0) * 2.0
        feature_weight = (
            rank_weight
            * float(feature.get("mean_shift", 0.0))
            * float(feature.get("exclusivity", 0.0))
            * before_strength
            * before_bonus
        )
        if feature_weight <= 0:
            continue
        for label, fraction in feature.get("top_label_histogram", {}).items():
            label_scores[int(label)] += feature_weight * float(fraction)

    label_scores = {label: score for label, score in label_scores.items() if score > 0}
    top_labels = _top_class_items(label_scores, class_names, top_k=getattr(args, "crossleak_top_labels", 3))
    return label_scores, top_labels


def _infer_crossleak_features(scored, labels, indices, class_names, args):
    latent_shift = scored["latent_shift"].numpy()
    rho = scored["rho"].numpy()
    exclusivity = scored["exclusivity"].numpy()
    z_before = scored["z_before"].numpy()
    labels_np = labels.numpy()
    indices_np = indices.numpy()

    top_feature_k = min(getattr(args, "crossleak_top_features", 10), latent_shift.shape[1])
    feature_strength = latent_shift.mean(axis=0) * exclusivity
    top_features = np.argsort(-feature_strength)[:top_feature_k]
    per_feature = []

    for feature_id in top_features:
        sample_scores = latent_shift[:, feature_id] * max(exclusivity[feature_id], 1e-12)
        top_sample_k = min(getattr(args, "crossleak_top_samples_per_feature", 12), len(sample_scores))
        top_sample_pos = np.argsort(-sample_scores)[:top_sample_k]
        top_labels = labels_np[top_sample_pos]
        top_label_hist = _label_histogram(top_labels, num_classes=10)
        dominant_label = max(top_label_hist, key=top_label_hist.get) if top_label_hist else None
        direction = "before_exclusive_deleted" if rho[feature_id] >= 0.5 else "after_exclusive_compensation"

        per_feature.append({
            "feature_id": int(feature_id),
            "direction": direction,
            "decoder_before_ratio": float(rho[feature_id]),
            "exclusivity": float(exclusivity[feature_id]),
            "mean_shift": float(latent_shift[:, feature_id].mean()),
            "feature_strength": float(feature_strength[feature_id]),
            "mean_before_activation": float(z_before[:, feature_id].mean()),
            "dominant_label": int(dominant_label) if dominant_label is not None else None,
            "dominant_class_name": class_names[int(dominant_label)] if dominant_label is not None and class_names else None,
            "top_label_histogram": top_label_hist,
            "top_public_indices": [int(i) for i in indices_np[top_sample_pos].tolist()],
        })

    return per_feature


def _safe_name(name):
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(name)).strip("_")


def _save_top_feature_activations(scored, labels, indices, feature_info, class_names, args, name):
    if not getattr(args, "crossleak_save_feature_activations", True):
        return None

    output_dir = os.path.join(args.results_dir, "crossleak_feature_activations")
    os.makedirs(output_dir, exist_ok=True)
    base = _safe_name(name)
    json_path = os.path.join(output_dir, f"{base}_top_feature_activations.json")
    csv_path = os.path.join(output_dir, f"{base}_top_feature_activations.csv")

    z_before = scored["z_before"].numpy()
    z_after = scored["z_after"].numpy()
    latent_shift = scored["latent_shift"].numpy()
    total_scores = scored["score"].numpy()
    crosscoder_scores = scored["crosscoder_score"].numpy()
    labels_np = labels.numpy()
    indices_np = indices.numpy()
    rows = []
    top_sample_k = getattr(args, "crossleak_save_top_activations_per_feature", getattr(args, "crossleak_top_samples_per_feature", 12))

    for feature in feature_info:
        feature_id = int(feature["feature_id"])
        sample_scores = latent_shift[:, feature_id] * max(float(feature.get("exclusivity", 0.0)), 1e-12)
        top_pos = np.argsort(-sample_scores)[:min(top_sample_k, len(sample_scores))]
        for rank, pos in enumerate(top_pos, start=1):
            label = int(labels_np[pos])
            rows.append({
                "feature_id": feature_id,
                "feature_direction": feature["direction"],
                "feature_decoder_before_ratio": float(feature["decoder_before_ratio"]),
                "feature_exclusivity": float(feature["exclusivity"]),
                "feature_mean_shift": float(feature["mean_shift"]),
                "rank": int(rank),
                "dataset_index": int(indices_np[pos]),
                "label": label,
                "class_name": class_names[label] if class_names else str(label),
                "z_before": float(z_before[pos, feature_id]),
                "z_after": float(z_after[pos, feature_id]),
                "abs_shift": float(latent_shift[pos, feature_id]),
                "feature_sample_score": float(sample_scores[pos]),
                "crossleak_score": float(total_scores[pos]),
                "crosscoder_score": float(crosscoder_scores[pos]),
            })

    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["feature_id"])
        writer.writeheader()
        writer.writerows(rows)

    return {
        "json": json_path,
        "csv": csv_path,
        "num_rows": len(rows),
    }


def _true_erased_label_summary(train_set, client_data_info):
    erase_indices = []
    for info in client_data_info.values():
        erase_indices.extend(info["erase_indices"])
    labels = []
    for idx in erase_indices:
        _, y = train_set[idx]
        labels.append(int(y.argmax().item()) if torch.is_tensor(y) and y.ndim > 0 else int(y))
    hist = _label_histogram(labels, num_classes=10)
    top_labels = sorted(hist.items(), key=lambda item: item[1], reverse=True)
    return {
        "num_erased": len(labels),
        "label_histogram": hist,
        "top_labels": [{"label": int(label), "fraction": float(frac)} for label, frac in top_labels],
    }


def _label_set_metrics(predicted_labels, true_erased, args):
    k = int(getattr(args, "unlearning_class_range", getattr(args, "crossleak_top_labels", 3)))
    k = max(1, min(k, 10))
    true_labels = [item["label"] for item in true_erased["top_labels"][:k]]
    pred_labels = [item["label"] for item in predicted_labels[:k]]
    true_set = set(true_labels)
    pred_set = set(pred_labels)
    intersection = true_set.intersection(pred_set)
    union = true_set.union(pred_set)
    precision = len(intersection) / len(pred_set) if pred_set else 0.0
    recall = len(intersection) / len(true_set) if true_set else 0.0
    jaccard = len(intersection) / len(union) if union else 0.0
    return {
        "eval_k": k,
        "true_labels": true_labels,
        "predicted_labels": pred_labels,
        "num_correct": len(intersection),
        "hit": len(intersection) > 0,
        "precision": float(precision),
        "recall": float(recall),
        "jaccard": float(jaccard),
        "exact_match": pred_set == true_set if true_set else False,
    }


def _compact_label_items(items):
    return [(item["class_name"], int(item["label"]), round(float(item["score"]), 6)) for item in items]


def crossleak_attack(before_model, after_model, public_dataset, train_set_for_eval, client_data_info, args, name='model'):
    """
    SA-CrossLeak attack under the intended threat model: the attack only uses
    pre/post aggregated global models plus a public labeled probe set. The true
    erased labels are used only after the attack for evaluation/debug printing.
    """
    class_names = getattr(public_dataset, "classes", [str(i) for i in range(10)])
    max_samples = getattr(args, "crossleak_max_samples", 1000)
    public_loader = DataLoader(
        IndexedDataset(public_dataset),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
    )

    acts = _collect_crossleak_activations(before_model, after_model, public_loader, args, max_samples=max_samples)
    crosscoder, mean, std = train_crosscoder(acts["before"], acts["after"], args)
    scored = _score_crossleak_samples(
        crosscoder,
        acts["before"],
        acts["after"],
        acts["logit_shift"],
        mean,
        std,
        args,
    )

    labels = acts["labels"]
    scores = scored["score"].numpy()
    class_scores = {}
    for label in range(10):
        mask = labels.numpy() == label
        if np.any(mask):
            class_scores[int(label)] = float(np.mean(scores[mask]))

    class_mean_predicted_unlearned_labels = _top_class_items(
        class_scores,
        class_names,
        top_k=getattr(args, "crossleak_top_labels", 3),
    )
    inferred_features = _infer_crossleak_features(scored, labels, acts["indices"], class_names, args)
    feature_vote_scores, feature_vote_predicted_unlearned_labels = _feature_vote_label_scores(
        inferred_features,
        class_names,
        args,
    )
    feature_activation_paths = _save_top_feature_activations(
        scored,
        labels,
        acts["indices"],
        inferred_features,
        class_names,
        args,
        name,
    )
    true_erased = _true_erased_label_summary(train_set_for_eval, client_data_info)
    feature_vote_label_metrics = _label_set_metrics(
        feature_vote_predicted_unlearned_labels,
        true_erased,
        args,
    )
    class_mean_label_metrics = _label_set_metrics(
        class_mean_predicted_unlearned_labels,
        true_erased,
        args,
    )

    result = {
        "attacker_uses_erased_data": False,
        "predicted_unlearned_labels": feature_vote_predicted_unlearned_labels,
        "feature_vote_predicted_unlearned_labels": feature_vote_predicted_unlearned_labels,
        "class_mean_predicted_unlearned_labels": class_mean_predicted_unlearned_labels,
        "class_scores": {
            int(label): {
                "class_name": class_names[int(label)] if class_names else str(label),
                "score": float(score),
            }
            for label, score in class_scores.items()
        },
        "feature_vote_label_scores": {
            int(label): {
                "class_name": class_names[int(label)] if class_names else str(label),
                "score": float(score),
            }
            for label, score in feature_vote_scores.items()
        },
        "inferred_unlearned_features": inferred_features,
        "feature_activation_paths": feature_activation_paths,
        "true_erased_label_summary_for_eval": true_erased,
        "top_label_hit": bool(feature_vote_label_metrics["hit"]),
        "feature_vote_top_label_hit": bool(feature_vote_label_metrics["hit"]),
        "class_mean_top_label_hit": bool(class_mean_label_metrics["hit"]),
        "feature_vote_label_metrics": feature_vote_label_metrics,
        "class_mean_label_metrics": class_mean_label_metrics,
        "num_public_probe_samples": int(len(labels)),
    }

    print(f"\nSA-CrossLeak results for {name}:")
    print(f"  true_labels={feature_vote_label_metrics['true_labels']} erased_hist={true_erased['label_histogram']}")
    print(f"  feature_vote_pred={_compact_label_items(feature_vote_predicted_unlearned_labels)}")
    print(
        "  feature_vote_metrics: "
        f"k={feature_vote_label_metrics['eval_k']}, "
        f"correct={feature_vote_label_metrics['num_correct']}, "
        f"precision={feature_vote_label_metrics['precision']:.4f}, "
        f"recall={feature_vote_label_metrics['recall']:.4f}, "
        f"jaccard={feature_vote_label_metrics['jaccard']:.4f}, "
        f"exact={feature_vote_label_metrics['exact_match']}"
    )
    print(
        "  class_mean_baseline: "
        f"pred={_compact_label_items(class_mean_predicted_unlearned_labels)}, "
        f"precision={class_mean_label_metrics['precision']:.4f}, "
        f"recall={class_mean_label_metrics['recall']:.4f}, "
        f"jaccard={class_mean_label_metrics['jaccard']:.4f}"
    )
    print("  top deletion-sensitive features:")
    for feature in inferred_features[:getattr(args, "crossleak_print_top_features", 3)]:
        print(
            f"    z{feature['feature_id']} {feature['direction']}, "
            f"rho={feature['decoder_before_ratio']:.3f}, "
            f"shift={feature['mean_shift']:.3f}, "
            f"dominant={feature['dominant_class_name']}, "
            f"hist={feature['top_label_histogram']}"
        )
    if feature_activation_paths is not None:
        print("  saved top feature activations:",
              feature_activation_paths["csv"],
              f"({feature_activation_paths['num_rows']} rows)")

    return result


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
    verbose_unlearning = getattr(args, "verbose_unlearning", False)

    if verbose_unlearning:
        print(len(erasing_dataset.dataset))
    for epoch in range(args.local_epochs):
        model.train()
        for (x_e, y_e), (x_r, y_r) in zip(erasing_dataset, dataloader_remaining_after_aux):
            x_e, y_e = x_e.to(args.device), y_e.to(args.device)  # (B, C, H, W), (B, 10)
            x_r, y_r = x_r.to(args.device), y_r.to(args.device)  # (B, C, H, W), (B, 10)

            logits_z_e, logits_y_e, x_hat_e, mu_e, logvar_e = model(x_e, mode='with_reconstruction')  # (B, C* h* w), (B, N, 10)

            logits_z_r, logits_y_r, x_hat_r, mu_r, logvar_r = model(x_r, mode='with_reconstruction')

            KLD_element = mu_e.pow(2).add_(logvar_e.exp()).mul_(-1).add_(1).add_(logvar_e).to(args.device)
            KLD_mean = torch.mean(KLD_element).mul_(-0.5).to(args.device)
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
            if verbose_unlearning and step % len(erasing_dataset) % 10000 == 0:
                print(f'[{epoch}/{0 + args.num_epochs}:{step % len(erasing_dataset):3d}] '
                      + ', '.join([f'{k} {v:.3f}' for k, v in metrics.items()]))
                x_cpu = x_e.cpu().data
                x_cpu = x_cpu.view(x_cpu.size(0), *args.x_shape)
                grid = torchvision.utils.make_grid(x_cpu, nrow=4)
                # plt.imshow(np.transpose(grid, (1, 2, 0)))  # 交换维度，从GBR换成RGB
                # plt.show()

                print(acc_back)
                print("print x grad")
                # print(updated_x)
            backdoor_acc = eva_vib(
                model,
                erasing_dataset,
                args,
                name='on erased data',
                epoch=999,
                verbose=verbose_unlearning,
            )
            model.train()
            if backdoor_acc < 0.8:
                break
    if verbose_unlearning:
        print("backdoor_acc_list", backdoor_acc_list)
    return model


def synchronize_device(args):
    if getattr(args, "device", None) is not None and args.device.type == "cuda":
        torch.cuda.synchronize(args.device)


def federated_unlearning_one_round(
    global_model: nn.Module,
    train_set,
    client_subsets,
    args,
    num_unlearn_clients=3,
    erase_ratio=0.05,
    base_seed=0,
    loss_fn=None,
    round_idx=1,
):
    """
    One federated unlearning round:
      - pick num_unlearn_clients clients
      - each does prepare_unl on 5% erase + 95% remain
      - in round 1, non-chosen clients do not upload
      - in later rounds, non-chosen clients continue normal local training
      - FedAvg aggregate uploaded client models

    Returns:
      new_global_model, chosen_client_ids, client_data_info, round_parallel_time
    where client_data_info[cid] stores indices and loaders for chosen clients.
    round_parallel_time is max(client local compute time) because clients run in parallel.
    """
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    reconstruction_function = nn.MSELoss(reduction='mean')

    rng = np.random.RandomState(base_seed)
    all_client_ids = list(range(len(client_subsets)))
    chosen_client_ids = rng.choice(all_client_ids, size=num_unlearn_clients, replace=False).tolist()
    print("Chosen clients for unlearning:", chosen_client_ids)

    client_weights = []
    client_data_info = {}  # store erase/remain indices and loaders for chosen clients
    client_times = []

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

            synchronize_device(args)
            client_start_time = time.time()
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

            synchronize_device(args)
            client_times.append(time.time() - client_start_time)
            client_weights.append(copy.deepcopy(local_model.state_dict()))
        else:
            synchronize_device(args)
            client_start_time = time.time()
            # --- build erase/remain subsets + loaders for this client ---
            erase_subset, remain_subset, erase_idx, remain_idx = split_client_indices(
                client_subsets[cid],
                erase_ratio=erase_ratio,
                seed=base_seed + 1000 + cid,  # make each client split deterministic but different
                target_classes=resolve_unlearning_target_classes(args),
            )

            erase_loader = make_loader(erase_subset, batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=True)
            remain_loader = make_loader(remain_subset, batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=True)
            erase_label_hist = _label_histogram(
                [_dataset_label_at(train_set, int(idx)) for idx in erase_idx],
                num_classes=10,
            )
            if getattr(args, "class_conditional_unlearning", False):
                print(f"Client {cid} class-conditional erase label histogram: {erase_label_hist}")

            client_data_info[cid] = {
                "erase_indices": erase_idx,         # indices into train_set
                "remain_indices": remain_idx,       # indices into train_set
                "erase_loader": erase_loader,
                "remain_loader": remain_loader,
                "erase_size": len(erase_subset),
                "remain_size": len(remain_subset),
                "erase_label_histogram": erase_label_hist,
            }

            local_model = prepare_unl(
                erasing_dataset=erase_loader,
                dataloader_remaining_after_aux=remain_loader,
                model=local_model,
                loss_fn=loss_fn,
                args=args,
                noise_flag="no noise",
            )

            synchronize_device(args)
            client_times.append(time.time() - client_start_time)
            client_weights.append(copy.deepcopy(local_model.state_dict()))

        del local_model
        torch.cuda.empty_cache()

    # --- aggregate ---
    new_sd = FedAvg(client_weights)
    new_global_model = copy.deepcopy(global_model)
    new_global_model.load_state_dict(new_sd)
    new_global_model.to(args.device)
    new_global_model.eval()
    round_parallel_time = max(client_times) if client_times else 0.0

    return new_global_model, chosen_client_ids, client_data_info, round_parallel_time


def build_erased_data_loader(train_set, client_data_info, args):
    erase_indices = []
    for info in client_data_info.values():
        erase_indices.extend(info["erase_indices"])
    if len(erase_indices) == 0:
        return None
    erase_subset = Subset(train_set, erase_indices)
    return make_loader(erase_subset, batch_size=args.batch_size, shuffle=False, num_workers=1)


def configure_cifar10_unlearning_args():
    args = args_parser()
    args.gpu = 0
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    args.iid = True
    args.dataset = 'CIFAR10'
    args.add_noise = False
    args.beta = 0.0001
    args.mse_rate = 0.1
    args.lr = 0.0001
    args.dimZ = 512
    args.batch_size = 200

    args.num_clients = 10
    args.num_unlearn_clients = 1
    args.unlearning_ratio = 0.05
    args.class_conditional_unlearning = True
    args.unlearn_target_classes = [7, 1, 9, 3, 5]
    args.unlearning_class_range = 2
    args.sweep_unlearning_class_range = True
    args.max_unlearning_class_range = 5
    args.global_rounds = 5
    args.local_epochs = 10
    args.frac = 1.0
    args.seeds = [0, 1, 2, 3, 4]
    args.crossleak_max_samples = 1000
    args.crossleak_latent_dim = 256
    args.crossleak_epochs = 200
    args.crossleak_lr = 0.001
    args.crossleak_l1_lambda = 0.001
    args.crossleak_feature_tau = 0.15
    args.crossleak_logit_weight = 0.1
    args.crossleak_top_labels = 3
    args.crossleak_top_features = 10
    args.crossleak_top_samples_per_feature = 12
    args.crossleak_before_exclusive_vote_bonus = 1.5
    args.crossleak_print_top_features = 3
    args.crossleak_save_feature_activations = True
    args.crossleak_save_top_activations_per_feature = 20
    args.verbose_unlearning = False
    args.results_dir = os.path.dirname(os.path.abspath(__file__))
    return args


def load_cifar10_data(args, seed):
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
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=1)
    client_subsets = split_dataset_iid(train_set, args.num_clients, seed=seed)
    args.x_shape = (3, 32, 32)
    return train_set, train_set_no_aug, test_set, train_loader, test_loader, client_subsets


def load_global_cifar10_model(args):
    ckpt_path = "global_vib_cifar10_fedavg.pt"
    ckpt = torch.load(ckpt_path, map_location=args.device)
    global_vib, _ = init_vib(args)
    global_vib.load_state_dict(ckpt["model_state"])
    global_vib.to(args.device)
    global_vib.eval()
    print("Loaded checkpoint from round:", ckpt.get("round", "unknown"))
    return global_vib, ckpt


def summarize_mia(prefix, mia_result):
    if mia_result is None:
        return f"{prefix}_mia_acc=nan, {prefix}_mia_auc=nan"
    return (
        f"{prefix}_mia_acc={mia_result['attack_train_acc']:.4f}, "
        f"{prefix}_mia_auc={mia_result['attack_auc']:.4f}"
    )


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
        "erased_acc": result["erased_acc"],
        "running_time": result["running_time"],
    }
    for prefix in ("mia_before", "mia_after"):
        mia_result = result.get(prefix)
        if mia_result is None:
            continue
        for key, value in mia_result.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                metrics[f"{prefix}_{key}"] = float(value)
    crossleak = result.get("crossleak")
    if crossleak is not None:
        metrics["crossleak_top_label_hit"] = float(bool(crossleak.get("top_label_hit", False)))
        metrics["crossleak_feature_vote_top_label_hit"] = float(bool(crossleak.get("feature_vote_top_label_hit", False)))
        metrics["crossleak_class_mean_top_label_hit"] = float(bool(crossleak.get("class_mean_top_label_hit", False)))
        for prefix, label_metrics in (
            ("crossleak_feature_vote", crossleak.get("feature_vote_label_metrics")),
            ("crossleak_class_mean", crossleak.get("class_mean_label_metrics")),
        ):
            if label_metrics is None:
                continue
            for key in ("eval_k", "num_correct", "precision", "recall", "jaccard", "exact_match"):
                value = label_metrics.get(key)
                if isinstance(value, (bool, np.bool_)):
                    metrics[f"{prefix}_{key}"] = float(value)
                elif isinstance(value, (int, float, np.integer, np.floating)):
                    metrics[f"{prefix}_{key}"] = float(value)
        predicted = crossleak.get("predicted_unlearned_labels", [])
        if predicted:
            metrics["crossleak_top1_label_score"] = float(predicted[0]["score"])
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
            group_key = (
                result.get("unlearning_class_range", 1),
                result["num_unlearn_clients"],
            )
            grouped.setdefault(group_key, []).append((seed, flatten_result_metrics(result)))

    summary = []
    for group_key in sorted(grouped):
        class_range, num_clients = group_key
        metric_names = sorted({name for _, metrics in grouped[group_key] for name in metrics})
        row = {
            "unlearning_class_range": class_range,
            "num_unlearn_clients": num_clients,
            "num_seeds": len(grouped[group_key]),
            "seeds": [seed for seed, _ in grouped[group_key]],
        }
        for metric_name in metric_names:
            values = [metrics.get(metric_name, float("nan")) for _, metrics in grouped[group_key]]
            metric_mean, metric_std = mean_std(values)
            row[f"{metric_name}_mean"] = metric_mean
            row[f"{metric_name}_std"] = metric_std
        summary.append(row)
    return summary


def print_seed_sweep_summary(summary):
    print("\n===== CIFAR10 RFU multi-seed mean/std summary =====")
    for row in summary:
        parts = [
            f"class_range={row.get('unlearning_class_range', 1)}",
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
    raw_path = os.path.join(args.results_dir, "rfu_multi_seed_raw.json")
    summary_json_path = os.path.join(args.results_dir, "rfu_multi_seed_summary.json")
    summary_csv_path = os.path.join(args.results_dir, "rfu_multi_seed_summary.csv")

    with open(raw_path, "w") as f:
        json.dump(all_seed_results, f, indent=2)
    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=2)

    fieldnames = sorted({key for row in summary for key in row})
    preferred = ["unlearning_class_range", "num_unlearn_clients", "num_seeds", "seeds"]
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
    print(f"\n===== CIFAR10 RFU seed={seed} =====")

    train_set, train_set_no_aug, test_set, train_loader, test_loader, client_subsets = load_cifar10_data(args, seed)
    global_vib, ckpt = load_global_cifar10_model(args)
    _ = eva_vib(global_vib, test_loader, args, name=f"loaded model (seed={seed})", epoch=ckpt.get("round", 0))

    loss_fn = nn.CrossEntropyLoss()
    sweep_results = []

    if getattr(args, "sweep_unlearning_class_range", False):
        sweep_items = [
            (args.num_unlearn_clients, class_range)
            for class_range in range(1, int(getattr(args, "max_unlearning_class_range", 5)) + 1)
        ]
    else:
        sweep_items = [
            (num_unlearn_clients, int(getattr(args, "unlearning_class_range", 1)))
            for num_unlearn_clients in range(1, 6)
        ]

    for num_unlearn_clients, class_range in sweep_items:
        args.num_unlearn_clients = num_unlearn_clients
        args.unlearning_class_range = class_range
        target_classes = resolve_unlearning_target_classes(args)
        print(
            f"\n===== RFU test: num_unlearn_clients={num_unlearn_clients}, "
            f"unlearning_class_range={class_range}, target_classes={target_classes} ====="
        )

        run_model = copy.deepcopy(global_vib)
        run_model.to(args.device)
        run_model.eval()
        pre_unlearn_vib = copy.deepcopy(run_model)
        pre_unlearn_vib.eval()

        running_time = 0.0
        client_data_info = {}
        chosen_ids = []
        new_global_vib = run_model
        for rnd in range(1, args.global_rounds + 1):
            new_global_vib, chosen_ids, client_data_info, round_parallel_time = federated_unlearning_one_round(
                global_model=new_global_vib,
                train_set=train_set,
                client_subsets=client_subsets,
                args=args,
                num_unlearn_clients=args.num_unlearn_clients,
                erase_ratio=args.unlearning_ratio,
                base_seed=seed,
                loss_fn=loss_fn,
                round_idx=rnd,
            )
            running_time += round_parallel_time
            print(f"Round {rnd} parallel client time: {round_parallel_time:.4f}s")

        acc_after = eva_vib(
            new_global_vib,
            test_loader,
            args,
            name=f"after federated unlearning ({num_unlearn_clients} clients)",
            epoch=args.global_rounds,
        )

        erased_loader = build_erased_data_loader(train_set_no_aug, client_data_info, args)
        erased_acc = float("nan")
        if erased_loader is not None:
            erased_acc = eva_vib(
                new_global_vib,
                erased_loader,
                args,
                name=f"erased data ({num_unlearn_clients} clients)",
                epoch=args.global_rounds,
            )

        mia_before = None
        mia_after = None
        mia_erase_loader, mia_retain_loader, mia_nonmember_loader = build_learned_mia_loaders(
            train_set=train_set_no_aug,
            test_set=test_set,
            client_data_info=client_data_info,
            args=args,
            seed=seed + 2024 + num_unlearn_clients,
        )
        if mia_erase_loader is not None:
            mia_before = learned_membership_inference_attack(
                pre_unlearn_vib,
                mia_erase_loader,
                mia_retain_loader,
                mia_nonmember_loader,
                args,
                name=f"before unlearning ({num_unlearn_clients} clients)",
            )
            mia_after = learned_membership_inference_attack(
                new_global_vib,
                mia_erase_loader,
                mia_retain_loader,
                mia_nonmember_loader,
                args,
                name=f"after federated unlearning ({num_unlearn_clients} clients)",
            )

        crossleak = crossleak_attack(
            before_model=pre_unlearn_vib,
            after_model=new_global_vib,
            public_dataset=test_set,
            train_set_for_eval=train_set_no_aug,
            client_data_info=client_data_info,
            args=args,
            name=f"RFU before/after ({num_unlearn_clients} clients, {class_range} classes)",
        )

        result = {
            "num_unlearn_clients": num_unlearn_clients,
            "unlearning_class_range": class_range,
            "unlearning_target_classes": target_classes,
            "seed": seed,
            "chosen_ids": chosen_ids,
            "acc_after": acc_after,
            "erased_acc": erased_acc,
            "mia_before": mia_before,
            "mia_after": mia_after,
            "crossleak": crossleak,
            "running_time": running_time,
        }
        sweep_results.append(result)

        print(
            "RFU/CrossLeak summary: "
            f"clients={num_unlearn_clients}, class_range={class_range}, "
            f"targets={target_classes}, acc={acc_after:.4f}, erased_acc={erased_acc:.4f}, "
            f"mia_auc_before={mia_before['attack_auc']:.4f}, "
            f"mia_auc_after={mia_after['attack_auc']:.4f}, "
            f"cl_precision={crossleak['feature_vote_label_metrics']['precision']:.4f}, "
            f"cl_recall={crossleak['feature_vote_label_metrics']['recall']:.4f}, "
            f"cl_jaccard={crossleak['feature_vote_label_metrics']['jaccard']:.4f}, "
            f"class_mean_jaccard={crossleak['class_mean_label_metrics']['jaccard']:.4f}"
        )

        del run_model, pre_unlearn_vib, new_global_vib
        torch.cuda.empty_cache()

    print(f"\n===== CIFAR10 RFU sweep summary (seed={seed}) =====")
    for result in sweep_results:
        print(
            f"clients={result['num_unlearn_clients']}, "
            f"class_range={result.get('unlearning_class_range', 1)}, "
            f"targets={result.get('unlearning_target_classes')}, "
            f"seed={result['seed']}, acc={result['acc_after']:.4f}, "
            f"erased_acc={result['erased_acc']:.4f}, "
            f"mia_auc_before={result['mia_before']['attack_auc']:.4f}, "
            f"mia_auc_after={result['mia_after']['attack_auc']:.4f}, "
            f"cl_precision={result['crossleak']['feature_vote_label_metrics']['precision']:.4f}, "
            f"cl_recall={result['crossleak']['feature_vote_label_metrics']['recall']:.4f}, "
            f"cl_jaccard={result['crossleak']['feature_vote_label_metrics']['jaccard']:.4f}, "
            f"class_mean_jaccard={result['crossleak']['class_mean_label_metrics']['jaccard']:.4f}"
        )

    return sweep_results


def test_unlearning_clients_sweep():
    args = configure_cifar10_unlearning_args()
    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))
    print(
        "CIFAR10 RFU/CrossLeak config: "
        f"device={args.device}, seeds={args.seeds}, "
        f"num_clients={args.num_clients}, num_unlearn_clients={args.num_unlearn_clients}, "
        f"unlearning_ratio={args.unlearning_ratio}, "
        f"class_conditional={args.class_conditional_unlearning}, "
        f"target_candidates={args.unlearn_target_classes}, "
        f"sweep_class_range={args.sweep_unlearning_class_range}, "
        f"max_class_range={args.max_unlearning_class_range}, "
        f"crossleak_probe={args.crossleak_max_samples}"
    )

    all_seed_results = []
    for seed in args.seeds:
        seed_results = run_unlearning_clients_sweep_for_seed(args, seed)
        all_seed_results.append({
            "seed": seed,
            "results": seed_results,
        })

    summary = summarize_seed_sweep(all_seed_results)
    print_seed_sweep_summary(summary)
    # save_seed_sweep_records(all_seed_results, summary, args)
    return all_seed_results, summary


if __name__ == "__main__":
    test_unlearning_clients_sweep()
