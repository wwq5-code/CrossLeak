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
    parser.add_argument('--dataset', choices=['MNIST', 'CIFAR10'], default='CIFAR10')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs for VIBI.')
    parser.add_argument('--explainer_type', choices=['Unet', 'ResNet_2x', 'ResNet_4x', 'ResNet_8x'],
                        default='ResNet_4x')
    parser.add_argument('--xpl_channels', type=int, choices=[1, 3], default=1)
    parser.add_argument('--k', type=int, default=12, help='Number of chunks.')
    parser.add_argument('--beta', type=float, default=0, help='beta in objective J = I(y,t) - beta * I(x,t).')
    parser.add_argument('--unlearning_ratio', type=float, default=0.1)
    parser.add_argument('--backdoor_dir', type=str, default='backdoored_cifar10')
    parser.add_argument('--num_samples', type=int, default=4,
                        help='Number of samples used for estimating expectation over p(t|x).')
    args = parser.parse_args()
    return args


class BackdooredImageFolder(Dataset):
    def __init__(self, root_dir, transform=None, default_label=7, image_mode='RGB'):
        self.root_dir = root_dir
        self.transform = transform or transforms.ToTensor()
        self.default_label = default_label
        self.image_mode = image_mode
        self.samples = []

        if os.path.isdir(root_dir):
            for name in sorted(os.listdir(root_dir)):
                if name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append(os.path.join(root_dir, name))

        if len(self.samples) == 0:
            raise FileNotFoundError(f"No backdoored images found in {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        image = Image.open(path).convert(self.image_mode)
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


def resolve_existing_path(*candidates):
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return candidates[0]


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
            sparse_loss = zb.abs().mean() + za.abs().mean()
            loss = recon_loss + l1_lambda * sparse_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    crosscoder.eval()
    return crosscoder, mean, std


@torch.no_grad()
def _score_crossleak_samples(crosscoder, h_before, h_after, logit_shift, mean, std, args, score_mean=None, score_std=None, logit_mean=None, logit_std=None):
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
    raw_crosscoder_score = (latent_shift * weights.unsqueeze(0)).sum(dim=1)
    logit_shift = logit_shift.float().to(args.device)

    if score_mean is None:
        score_mean = raw_crosscoder_score.mean()
    if score_std is None:
        score_std = raw_crosscoder_score.std().clamp_min(1e-6)
    if logit_mean is None:
        logit_mean = logit_shift.mean()
    if logit_std is None:
        logit_std = logit_shift.std().clamp_min(1e-6)

    crosscoder_score = (raw_crosscoder_score - score_mean) / score_std
    normalized_logit_shift = (logit_shift - logit_mean) / logit_std
    total_score = crosscoder_score + getattr(args, "crossleak_logit_weight", 0.1) * normalized_logit_shift

    return {
        "z_before": z_before.cpu(),
        "z_after": z_after.cpu(),
        "latent_shift": latent_shift.cpu(),
        "score": total_score.cpu(),
        "crosscoder_score": crosscoder_score.cpu(),
        "raw_crosscoder_score": raw_crosscoder_score.cpu(),
        "rho": rho.cpu(),
        "exclusivity": exclusivity.cpu(),
        "feature_mask": feature_mask.cpu(),
        "score_mean": score_mean.detach().cpu(),
        "score_std": score_std.detach().cpu(),
        "logit_mean": logit_mean.detach().cpu(),
        "logit_std": logit_std.detach().cpu(),
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
            "mean_before_activation": float(z_before[:, feature_id].mean()),
            "dominant_label": int(dominant_label) if dominant_label is not None else None,
            "dominant_class_name": class_names[int(dominant_label)] if dominant_label is not None and class_names else None,
            "top_label_histogram": top_label_hist,
            "top_public_indices": [int(i) for i in indices_np[top_sample_pos].tolist()],
        })

    return per_feature


def _feature_vote_label_scores(public_feature_info, probe_feature_info, class_names, args):
    """
    Infer deleted labels from shifted features instead of class-average sample
    scores. Probe top features vote for labels according to their public
    top-activating label histograms.
    """
    if not probe_feature_info:
        return {}, []
    public_by_id = {feature["feature_id"]: feature for feature in public_feature_info}
    before_bonus = getattr(args, "crossleak_before_exclusive_vote_bonus", 1.5)
    label_scores = {label: 0.0 for label in range(10)}

    for rank, probe_feature in enumerate(probe_feature_info):
        public_feature = public_by_id.get(probe_feature["feature_id"])
        if public_feature is None:
            continue
        rank_weight = 1.0 / float(rank + 1)
        feature_weight = (
            rank_weight
            * float(probe_feature.get("mean_shift", 0.0))
            * max(float(probe_feature.get("decoder_before_ratio", 0.5)) - 0.5, 0.0) * 2.0
        )
        if probe_feature.get("direction") == "before_exclusive_deleted":
            feature_weight *= before_bonus
        if feature_weight <= 0:
            continue
        for label, fraction in public_feature.get("top_label_histogram", {}).items():
            label_scores[int(label)] += feature_weight * float(fraction)

    label_scores = {label: score for label, score in label_scores.items() if score > 0}
    top_labels = _top_class_items(label_scores, class_names, top_k=getattr(args, "crossleak_top_labels", 3))
    return label_scores, top_labels


def _true_erased_label_summary(train_set, client_data_info, args):
    erase_indices = []
    for info in client_data_info.values():
        erase_indices.extend(info["erase_indices"])
    labels = []
    backdoor_count = 0
    for idx in erase_indices:
        if isinstance(idx, str) and idx.startswith("backdoor:"):
            labels.append(int(args.backdoor_target))
            backdoor_count += 1
            continue
        _, y = train_set[idx]
        labels.append(int(y.argmax().item()) if torch.is_tensor(y) and y.ndim > 0 else int(y))
    hist = _label_histogram(labels, num_classes=10)
    top_labels = sorted(hist.items(), key=lambda item: item[1], reverse=True)
    return {
        "num_erased": len(labels),
        "num_backdoor_erased": backdoor_count,
        "label_histogram": hist,
        "top_labels": [{"label": int(label), "fraction": float(frac)} for label, frac in top_labels],
    }


def _score_extra_probe_dataset(before_model, after_model, crosscoder, mean, std, reference_scored, probe_dataset, args):
    probe_loader = DataLoader(
        IndexedDataset(probe_dataset),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
    )
    acts = _collect_crossleak_activations(before_model, after_model, probe_loader, args, max_samples=None)
    scored = _score_crossleak_samples(
        crosscoder,
        acts["before"],
        acts["after"],
        acts["logit_shift"],
        mean,
        std,
        args,
        score_mean=reference_scored["score_mean"].to(args.device),
        score_std=reference_scored["score_std"].to(args.device),
        logit_mean=reference_scored["logit_mean"].to(args.device),
        logit_std=reference_scored["logit_std"].to(args.device),
    )
    top_feature_k = min(getattr(args, "crossleak_top_features", 10), scored["latent_shift"].shape[1])
    feature_strength = scored["latent_shift"].mean(dim=0).numpy() * scored["exclusivity"].numpy()
    top_features = np.argsort(-feature_strength)[:top_feature_k]
    return {
        "num_samples": int(len(acts["labels"])),
        "label_histogram": _label_histogram(acts["labels"].numpy(), num_classes=10),
        "mean_score": float(scored["score"].mean().item()),
        "mean_crosscoder_score": float(scored["crosscoder_score"].mean().item()),
        "top_shifted_features": [
            {
                "feature_id": int(fid),
                "direction": "before_exclusive_deleted" if float(scored["rho"][fid]) >= 0.5 else "after_exclusive_compensation",
                "decoder_before_ratio": float(scored["rho"][fid]),
                "exclusivity": float(scored["exclusivity"][fid]),
                "mean_shift": float(scored["latent_shift"][:, fid].mean().item()),
                "feature_strength": float(feature_strength[fid]),
            }
            for fid in top_features
        ],
    }


def crossleak_attack(before_model, after_model, public_dataset, train_set_for_eval, client_data_info, args, name='model', extra_probe_dataset=None, extra_probe_name='extra_probe'):
    """
    SA-CrossLeak attack under the intended threat model: train CrossCoder from
    pre/post aggregated global models plus a clean public probe set. If an
    extra probe set is provided, it is only scored after CrossCoder training,
    which is useful for checking whether backdoor-triggered samples exhibit a
    stronger deletion fingerprint.
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
    true_erased = _true_erased_label_summary(train_set_for_eval, client_data_info, args)
    true_label_set = {item["label"] for item in true_erased["top_labels"][:getattr(args, "crossleak_top_labels", 3)]}
    class_mean_pred_label_set = {item["label"] for item in class_mean_predicted_unlearned_labels}
    class_mean_top_label_hit = len(true_label_set.intersection(class_mean_pred_label_set)) > 0 if true_label_set else False

    extra_probe_result = None
    if extra_probe_dataset is not None:
        extra_probe_result = _score_extra_probe_dataset(
            before_model,
            after_model,
            crosscoder,
            mean,
            std,
            scored,
            extra_probe_dataset,
            args,
        )
    feature_vote_scores = {}
    feature_vote_predicted_unlearned_labels = []
    if extra_probe_result is not None:
        feature_vote_scores, feature_vote_predicted_unlearned_labels = _feature_vote_label_scores(
            inferred_features,
            extra_probe_result["top_shifted_features"],
            class_names,
            args,
        )
    feature_vote_pred_label_set = {item["label"] for item in feature_vote_predicted_unlearned_labels}
    feature_vote_top_label_hit = len(true_label_set.intersection(feature_vote_pred_label_set)) > 0 if true_label_set else False

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
        "true_erased_label_summary_for_eval": true_erased,
        "top_label_hit": bool(feature_vote_top_label_hit),
        "feature_vote_top_label_hit": bool(feature_vote_top_label_hit),
        "class_mean_top_label_hit": bool(class_mean_top_label_hit),
        "num_public_probe_samples": int(len(labels)),
        extra_probe_name: extra_probe_result,
    }

    print(f"\nSA-CrossLeak results for {name}:")
    print("  feature-vote deleted labels:",
          [(item["class_name"], round(item["score"], 4)) for item in feature_vote_predicted_unlearned_labels])
    print("  class-mean deleted labels:",
          [(item["class_name"], round(item["score"], 4)) for item in class_mean_predicted_unlearned_labels])
    print("  true erased label histogram:", true_erased["label_histogram"])
    print("  erased backdoor probe count:", true_erased["num_backdoor_erased"])
    print("  feature-vote top-label hit:", result["feature_vote_top_label_hit"])
    print("  class-mean top-label hit:", result["class_mean_top_label_hit"])
    print("  top inferred deleted/shifted features:")
    for feature in inferred_features[:5]:
        print(
            f"    z{feature['feature_id']} {feature['direction']}, "
            f"rho={feature['decoder_before_ratio']:.3f}, "
            f"dominant={feature['dominant_class_name']}, "
            f"hist={feature['top_label_histogram']}"
        )
    if extra_probe_result is not None:
        print(f"  {extra_probe_name} mean_score={extra_probe_result['mean_score']:.4f}, "
              f"mean_crosscoder_score={extra_probe_result['mean_crosscoder_score']:.4f}, "
              f"labels={extra_probe_result['label_histogram']}")
        print(f"  {extra_probe_name} top shifted features:",
              extra_probe_result["top_shifted_features"][:5])

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
            if step % len(erasing_dataset) % 10000 == 0:
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


def configure_backdoor_rfu_args():
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
    args.backdoor_dir = "/home/wwq/PerFU/PerFU/On_CIFAR10/backdoored_cifar10"
    args.backdoor_target = 7

    args.num_unlearn_clients = 1
    args.num_clients = 10
    args.global_rounds = 10
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
    args.results_dir = os.path.dirname(os.path.abspath(__file__))
    return args


def set_random_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_backdoor_data(args, seed):
    script_dir = os.path.dirname(__file__)
    if args.dataset == 'MNIST':
        trans_mnist = transforms.Compose([transforms.ToTensor()])
        train_set = MNIST('/home/wwq/Data/data/mnist', train=True, transform=trans_mnist, download=True)
        test_set = MNIST('/home/wwq/Data/data/mnist', train=False, transform=trans_mnist, download=False)
        train_set_no_aug = train_set
        backdoor_dir = args.backdoor_dir
        if not os.path.isabs(backdoor_dir):
            backdoor_dir = os.path.join(script_dir, backdoor_dir)
        backdoor_dataset = BackdooredImageFolder(
            backdoor_dir,
            transform=trans_mnist,
            default_label=args.backdoor_target,
            image_mode='L',
        )
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
        backdoor_dir = args.backdoor_dir
        if not os.path.isabs(backdoor_dir):
            backdoor_dir = os.path.join(script_dir, backdoor_dir)
        backdoor_dataset = BackdooredImageFolder(
            backdoor_dir,
            transform=test_transform,
            default_label=args.backdoor_target,
            image_mode='RGB',
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=1)
    backdoor_loader = DataLoader(backdoor_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)
    client_subsets = split_dataset_iid(train_set, args.num_clients, seed=seed)
    args.x_shape = (1, 28, 28) if args.dataset == 'MNIST' else (3, 32, 32)
    print(f"Loaded {len(backdoor_dataset)} backdoored {args.dataset} samples from {backdoor_dir}")
    return train_set, train_set_no_aug, test_set, train_loader, test_loader, backdoor_dataset, backdoor_loader, client_subsets


def load_backdoored_global_model(args):
    script_dir = os.path.dirname(__file__)
    ckpt_path = resolve_existing_path(
        os.path.join(script_dir, "global_vib_cifar10_fedavg_backdoored.pt"),
        "global_vib_cifar10_fedavg_backdoored.pt",
    )
    ckpt = torch.load(ckpt_path, map_location=args.device)
    global_vib, _ = init_vib(args)
    global_vib.load_state_dict(ckpt["model_state"])
    global_vib.to(args.device)
    global_vib.eval()
    print("Loaded checkpoint from round:", ckpt.get("round", "unknown"))
    return global_vib, ckpt


def run_backdoor_rfu_for_seed(args, seed):
    set_random_seed(seed)
    print(f"\n===== CIFAR10 backdoor RFU seed={seed} =====")
    train_set, train_set_no_aug, test_set, train_loader, test_loader, backdoor_dataset, backdoor_loader, client_subsets = load_backdoor_data(args, seed)
    global_vib, ckpt = load_backdoored_global_model(args)

    clean_acc_before = eva_vib(global_vib, test_loader, args, name=f"loaded model seed={seed}", epoch=ckpt.get("round", 0))
    backdoor_acc_before = eva_vib(global_vib, backdoor_loader, args, name=f"loaded model backdoor attack seed={seed}", epoch=ckpt.get("round", 0))
    pre_unlearn_vib = copy.deepcopy(global_vib)
    pre_unlearn_vib.eval()

    loss_fn = nn.CrossEntropyLoss()
    start_time = time.time()
    client_data_info = {}
    chosen_ids = []
    new_global_vib = global_vib

    for rnd in range(1, args.global_rounds + 1):
        new_global_vib, chosen_ids, client_data_info = federated_unlearning_one_round(
            global_model=new_global_vib,
            train_set=train_set,
            client_subsets=client_subsets,
            args=args,
            num_unlearn_clients=args.num_unlearn_clients,
            erase_ratio=0.05,
            base_seed=seed,
            loss_fn=loss_fn,
            backdoor_dataset=backdoor_dataset,
            round_idx=rnd,
        )
        eva_vib(new_global_vib, test_loader, args, name=f"after federated unlearning round {rnd}", epoch=rnd)
        eva_vib(new_global_vib, backdoor_loader, args, name=f"after federated unlearning round {rnd} backdoor attack", epoch=rnd)

    running_time = time.time() - start_time
    clean_acc_after = eva_vib(new_global_vib, test_loader, args, name="after federated unlearning", epoch=0)
    backdoor_acc_after = eva_vib(new_global_vib, backdoor_loader, args, name="after federated unlearning backdoor attack", epoch=0)

    crossleak_result = crossleak_attack(
        before_model=pre_unlearn_vib,
        after_model=new_global_vib,
        public_dataset=test_set,
        train_set_for_eval=train_set_no_aug,
        client_data_info=client_data_info,
        args=args,
        name=f"backdoor RFU before/after seed={seed}",
        extra_probe_dataset=backdoor_dataset,
        extra_probe_name="backdoor_probe",
    )

    for cid in chosen_ids:
        info = client_data_info[cid]
        print(f"\nClient {cid}: erase={info['erase_size']} remain={info['remain_size']}")
        print("First 10 erase indices:", info["erase_indices"][:10])
        print("First 10 remain indices:", info["remain_indices"][:10])

    result = {
        "seed": seed,
        "chosen_ids": chosen_ids,
        "clean_acc_before": clean_acc_before,
        "clean_acc_after": clean_acc_after,
        "backdoor_acc_before": backdoor_acc_before,
        "backdoor_acc_after": backdoor_acc_after,
        "running_time": running_time,
        "crossleak": crossleak_result,
        "num_unlearn_clients": args.num_unlearn_clients,
    }
    print(
        "Backdoor RFU/CrossLeak result: "
        f"seed={seed}, chosen_ids={chosen_ids}, "
        f"clean_acc_before={clean_acc_before:.4f}, clean_acc_after={clean_acc_after:.4f}, "
        f"backdoor_acc_before={backdoor_acc_before:.4f}, backdoor_acc_after={backdoor_acc_after:.4f}, "
        f"feature_vote_hit={crossleak_result['feature_vote_top_label_hit']}, "
        f"class_mean_hit={crossleak_result['class_mean_top_label_hit']}, "
        f"backdoor_probe_score={crossleak_result['backdoor_probe']['mean_score']:.4f}, "
        f"running_time={running_time:.4f}s"
    )

    del global_vib, pre_unlearn_vib, new_global_vib
    torch.cuda.empty_cache()
    return result


def mean_std(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return float(np.mean(arr)), std


def summarize_backdoor_results(results):
    metric_getters = {
        "clean_acc_before": lambda r: r["clean_acc_before"],
        "clean_acc_after": lambda r: r["clean_acc_after"],
        "backdoor_acc_before": lambda r: r["backdoor_acc_before"],
        "backdoor_acc_after": lambda r: r["backdoor_acc_after"],
        "feature_vote_top_label_hit": lambda r: float(r["crossleak"]["feature_vote_top_label_hit"]),
        "class_mean_top_label_hit": lambda r: float(r["crossleak"]["class_mean_top_label_hit"]),
        "backdoor_probe_mean_score": lambda r: r["crossleak"]["backdoor_probe"]["mean_score"],
        "backdoor_probe_mean_crosscoder_score": lambda r: r["crossleak"]["backdoor_probe"]["mean_crosscoder_score"],
        "running_time": lambda r: r["running_time"],
    }
    summary = {}
    for name, getter in metric_getters.items():
        values = [getter(result) for result in results]
        metric_mean, metric_std = mean_std(values)
        summary[f"{name}_mean"] = metric_mean
        summary[f"{name}_std"] = metric_std
    return summary


def print_backdoor_summary(summary, args):
    print("\n===== CIFAR10 backdoor RFU multi-seed mean/std summary =====")
    parts = [
        f"num_unlearn_clients={args.num_unlearn_clients}",
        f"num_seeds={len(args.seeds)}",
        f"seeds={args.seeds}",
    ]
    for key in sorted(summary):
        if key.endswith("_mean"):
            base = key[:-5]
            std_key = f"{base}_std"
            if std_key in summary:
                parts.append(f"{base}={summary[key]:.4f}±{summary[std_key]:.4f}")
    print(", ".join(parts))


def test_backdoor_rfu_multi_seed():
    args = configure_backdoor_rfu_args()
    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))
    print("device", args.device)
    print("multi-seed CIFAR10 backdoor RFU evaluation seeds:", args.seeds)

    results = []
    for seed in args.seeds:
        results.append(run_backdoor_rfu_for_seed(args, seed))

    summary = summarize_backdoor_results(results)
    print_backdoor_summary(summary, args)
    return results, summary


if __name__ == "__main__":
    test_backdoor_rfu_multi_seed()
