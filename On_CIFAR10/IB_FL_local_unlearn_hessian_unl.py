import sys

sys.argv = [""]
del sys

import argparse
import copy
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision.datasets import CIFAR10


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["CIFAR10"], default="CIFAR10")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--beta", type=float, default=0.0001)
    parser.add_argument("--unlearning_ratio", type=float, default=0.05)
    parser.add_argument("--mia_max_samples", type=int, default=1000)
    parser.add_argument("--mia_attack_epochs", type=int, default=200)
    parser.add_argument("--mia_attack_lr", type=float, default=0.01)
    parser.add_argument("--num_samples", type=int, default=4)
    return parser.parse_args()


class LinearModel(nn.Module):
    def __init__(self, n_feature=192, h_dim=3 * 32, n_output=10):
        super().__init__()
        self.fc1 = nn.Linear(n_feature, h_dim)
        self.fc2 = nn.Linear(h_dim, n_output)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


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
    def __init__(self, in_channels, block_features, num_classes=10):
        super().__init__()
        block_features = [block_features[0]] + block_features
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, block_features[0], kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(block_features[0]),
        )
        self.res_blocks = nn.ModuleList(
            [ResBlock(block_features[i], block_features[i + 1]) for i in range(len(block_features) - 1)]
        )
        self.linear_head = nn.Linear(block_features[-1], num_classes)

    def forward(self, x):
        x = self.expand(x)
        for res_block in self.res_blocks:
            x = res_block(x)
        x = F.avg_pool2d(x, x.shape[-1])
        return self.linear_head(x.reshape(x.shape[0], -1))


def resnet18(in_channels, num_classes):
    block_features = [64] * 2 + [128] * 2 + [256] * 2 + [512] * 2
    return ResNet(in_channels, block_features, num_classes)


class VIB(nn.Module):
    def __init__(self, encoder, approximator, decoder, reconstruction_dim):
        super().__init__()
        self.encoder = encoder
        self.approximator = approximator
        self.decoder = decoder
        self.fc3 = nn.Linear(reconstruction_dim, reconstruction_dim)

    def explain(self, x, mode="distribution"):
        double_logits_z = self.encoder(x)
        dim_z = double_logits_z.shape[1] // 2
        mu = double_logits_z[:, :dim_z].to(x.device)
        logvar = torch.log(torch.nn.functional.softplus(double_logits_z[:, dim_z:]).pow(2)).to(x.device)
        logits_z = self.reparametrize(mu, logvar)
        return logits_z, mu, logvar

    def forward(self, x, mode="distribution"):
        logits_z, mu, logvar = self.explain(x, mode="distribution")
        logits_y = self.approximator(logits_z).reshape((x.size(0), -1))
        if mode == "with_reconstruction":
            x_hat = self.reconstruction(logits_z)
            return logits_z, logits_y, x_hat, mu, logvar
        return logits_z, logits_y, mu, logvar

    def reconstruction(self, logits_z):
        output_x = self.decoder(logits_z.reshape((logits_z.size(0), -1)))
        return torch.sigmoid(self.fc3(output_x))

    @staticmethod
    def reparametrize(mu, logvar):
        std = logvar.mul(0.5).exp_()
        return torch.randn_like(std).mul(std).add_(mu)


def init_vib(args):
    reconstruction_dim = 3 * 32 * 32
    approximator = LinearModel(n_feature=args.dimZ, n_output=args.num_classes)
    encoder = resnet18(3, args.dimZ * 2)
    decoder = LinearModel(n_feature=args.dimZ, n_output=reconstruction_dim)
    vib = VIB(encoder, approximator, decoder, reconstruction_dim)
    vib.to(args.device)
    return vib


@torch.no_grad()
def eva_vib(vib, dataloader, args, name):
    vib.eval()
    correct = 0
    total = 0
    for x, y in dataloader:
        x, y = x.to(args.device), y.to(args.device)
        _, logits_y, _, _, _ = vib(x, mode="with_reconstruction")
        correct += (logits_y.argmax(dim=1) == y).sum().item()
        total += len(x)
    acc = correct / max(total, 1)
    print(f"{name} model acc: {acc:.4f}")
    return acc


def split_dataset_iid(dataset, num_clients, seed=0):
    rng = np.random.RandomState(seed)
    indices = np.arange(len(dataset))
    rng.shuffle(indices)
    return [Subset(dataset, split.tolist()) for split in np.array_split(indices, num_clients)]


def dataset_label_at(dataset, idx):
    _, y = dataset[idx]
    return int(y.argmax().item()) if torch.is_tensor(y) and y.ndim > 0 else int(y)


def resolve_unlearning_target_classes(args):
    class_range = int(getattr(args, "unlearning_class_range", 1))
    class_range = max(1, min(class_range, len(args.unlearn_target_classes)))
    return [int(label) for label in args.unlearn_target_classes[:class_range]]


def split_client_indices(client_subset, erase_ratio, seed, target_classes=None):
    rng = np.random.RandomState(seed)
    all_indices = list(client_subset.indices)
    erase_size = max(1, int(len(all_indices) * erase_ratio))
    target_classes = set(target_classes or [])

    if target_classes:
        target_indices = [
            idx for idx in all_indices
            if dataset_label_at(client_subset.dataset, idx) in target_classes
        ]
        rng.shuffle(target_indices)
        if len(target_indices) >= erase_size:
            erase_indices = target_indices[:erase_size]
        else:
            non_target_indices = [idx for idx in all_indices if idx not in set(target_indices)]
            rng.shuffle(non_target_indices)
            erase_indices = target_indices + non_target_indices[:erase_size - len(target_indices)]
    else:
        rng.shuffle(all_indices)
        erase_indices = all_indices[:erase_size]

    erase_set = set(erase_indices)
    remain_indices = [idx for idx in all_indices if idx not in erase_set]
    return erase_indices, remain_indices


def make_loader(dataset, batch_size, shuffle=True, drop_last=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1, drop_last=drop_last)


def cycle_loader(loader):
    while True:
        for batch in loader:
            yield batch


def fedavg(state_dicts):
    averaged = copy.deepcopy(state_dicts[0])
    for key in averaged:
        if torch.is_floating_point(averaged[key]):
            for state_dict in state_dicts[1:]:
                averaged[key] += state_dict[key].to(averaged[key].device)
            averaged[key] /= len(state_dicts)
    return averaged


def vib_loss(model, x, y, loss_fn, args):
    _, logits_y, _, mu, logvar = model(x, mode="with_reconstruction")
    kld = torch.mean(mu.pow(2).add(logvar.exp()).mul(-1).add(1).add(logvar)).mul(-0.5)
    return args.beta * kld + loss_fn(logits_y, y)


def train_local_vib(model, loader, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(args.local_epochs):
        for x, y in loader:
            x, y = x.to(args.device), y.to(args.device)
            loss = vib_loss(model, x, y, loss_fn, args)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def accumulate_named_grads_and_hessian(model, loader, loss_fn, args, max_batches=None, estimate_hessian=False):
    named_params = [
        (name, param)
        for name, param in model.named_parameters()
        if param.requires_grad
    ]
    params = [param for _, param in named_params]
    grad_sums = {
        name: torch.zeros_like(param, device=args.device)
        for name, param in named_params
    }
    hessian_sums = {
        name: torch.zeros_like(param, device=args.device)
        for name, param in named_params
    }
    num_batches = 0
    hessian_samples = int(getattr(args, "hessian_hutchinson_samples", 1))
    model.train()
    for batch_idx, (x, y) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x, y = x.to(args.device), y.to(args.device)
        loss = vib_loss(model, x, y, loss_fn, args)
        model.zero_grad(set_to_none=True)
        grads = torch.autograd.grad(
            loss,
            params,
            create_graph=estimate_hessian,
            retain_graph=estimate_hessian,
            allow_unused=True,
        )
        active = [
            (name, param, grad)
            for (name, param), grad in zip(named_params, grads)
            if grad is not None
        ]
        if not active:
            continue
        for name, _, grad in active:
            grad_sums[name].add_(grad.detach())
        if estimate_hessian:
            active_names = [name for name, _, _ in active]
            active_params = [param for _, param, _ in active]
            active_grads = [grad for _, _, grad in active]
            for sample_idx in range(hessian_samples):
                zs = [
                    torch.empty_like(param).bernoulli_(0.5).mul_(2).sub_(1)
                    for param in active_params
                ]
                hzs = torch.autograd.grad(
                    active_grads,
                    active_params,
                    grad_outputs=zs,
                    retain_graph=sample_idx < hessian_samples - 1,
                    allow_unused=True,
                )
                for name, z, hz in zip(active_names, zs, hzs):
                    if hz is None:
                        continue
                    hessian_sums[name].add_((hz.detach() * z).abs() / max(hessian_samples, 1))
        else:
            for name, _, grad in active:
                hessian_sums[name].add_(grad.detach().pow(2))
        num_batches += 1
    model.zero_grad(set_to_none=True)
    if num_batches == 0:
        return grad_sums, hessian_sums, 0
    for name in grad_sums:
        grad_sums[name].div_(num_batches)
        hessian_sums[name].div_(num_batches)
    return grad_sums, hessian_sums, num_batches


def accumulate_named_grads(model, loader, loss_fn, args, max_batches=None):
    grad_sums, _, num_batches = accumulate_named_grads_and_hessian(
        model,
        loader,
        loss_fn,
        args,
        max_batches=max_batches,
        estimate_hessian=False,
    )
    return grad_sums, num_batches


def accumulate_retained_hessian(model, loader, loss_fn, args, max_batches=None):
    return accumulate_named_grads_and_hessian(
        model,
        loader,
        loss_fn,
        args,
        max_batches=max_batches,
        estimate_hessian=True,
    )


def prepare_hessian_unlearning(erase_loader, remain_loader, model, loss_fn, args):
    erase_grads, erase_batches = accumulate_named_grads(
        model,
        erase_loader,
        loss_fn,
        args,
        max_batches=getattr(args, "hessian_max_erase_batches", None),
    )
    retain_grads, retain_hessian, retain_batches = accumulate_retained_hessian(
        model,
        remain_loader,
        loss_fn,
        args,
        max_batches=getattr(args, "hessian_max_retain_batches", None),
    )
    if erase_batches == 0 or retain_batches == 0:
        model.eval()
        return model

    damping = float(getattr(args, "hessian_damping", 1e-1))
    unlearn_lr = float(getattr(args, "hessian_unlearn_lr", 0.001))
    retain_lr = float(getattr(args, "hessian_retain_lr", 0.0005))
    max_update_norm = float(getattr(args, "hessian_max_update_norm", 0.01))

    with torch.no_grad():
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            inv_diag = 1.0 / (retain_hessian[name] + damping)
            update = unlearn_lr * erase_grads[name] * inv_diag
            if retain_lr > 0:
                update.sub_(retain_lr * retain_grads[name] * inv_diag)
            update = torch.clamp(update, min=-max_update_norm, max=max_update_norm)
            param.add_(update)

    model.zero_grad(set_to_none=True)
    model.eval()
    return model


def federated_unlearning_one_round(global_model, train_set, client_subsets, args, base_seed, round_idx):
    rng = np.random.RandomState(base_seed + round_idx)
    chosen_ids = rng.choice(np.arange(len(client_subsets)), size=args.num_unlearn_clients, replace=False).tolist()
    client_weights = []
    client_data_info = {}
    loss_fn = nn.CrossEntropyLoss()
    target_classes = resolve_unlearning_target_classes(args)

    for cid, client_subset in enumerate(client_subsets):
        local_model = copy.deepcopy(global_model).to(args.device)
        if cid not in chosen_ids:
            loader = make_loader(client_subset, args.batch_size, shuffle=True, drop_last=True)
            local_model = train_local_vib(local_model, loader, args)
        else:
            erase_indices, remain_indices = split_client_indices(
                client_subset,
                args.unlearning_ratio,
                seed=base_seed * 1000 + round_idx * 100 + cid,
                target_classes=target_classes,
            )
            erase_loader = make_loader(Subset(train_set, erase_indices), args.batch_size, shuffle=True, drop_last=False)
            remain_loader = make_loader(Subset(train_set, remain_indices), args.batch_size, shuffle=True, drop_last=True)
            local_model = prepare_hessian_unlearning(erase_loader, remain_loader, local_model, loss_fn, args)
            client_data_info[cid] = {
                "erase_indices": erase_indices,
                "remain_indices": remain_indices,
                "erase_size": len(erase_indices),
                "remain_size": len(remain_indices),
                "unlearning_steps": 1,
                "target_classes": target_classes,
                "unlearning_method": "diagonal_hessian_fisher",
            }

        client_weights.append(copy.deepcopy(local_model.state_dict()))
        del local_model

    new_global_model = copy.deepcopy(global_model)
    new_global_model.load_state_dict(fedavg(client_weights))
    new_global_model.to(args.device)
    new_global_model.eval()
    return new_global_model, chosen_ids, client_data_info


def build_erased_data_loader(train_set, client_data_info, args):
    erase_indices = []
    for info in client_data_info.values():
        erase_indices.extend(info["erase_indices"])
    return make_loader(Subset(train_set, erase_indices), args.batch_size, shuffle=False)


def build_learned_mia_loaders(train_set, test_set, client_data_info, args, seed=0):
    erase_indices = []
    retain_indices = []
    for info in client_data_info.values():
        erase_indices.extend(info["erase_indices"])
        retain_indices.extend(info["remain_indices"])

    if not erase_indices or not retain_indices:
        raise ValueError(
            "Cannot build MIA loaders because erased/retained indices are empty: "
            f"num_erased={len(erase_indices)}, num_retained={len(retain_indices)}. "
            "Check that at least one unlearning round has run."
        )

    rng = np.random.RandomState(seed)
    rng.shuffle(retain_indices)
    nonmember_indices = np.arange(len(test_set))
    rng.shuffle(nonmember_indices)

    n_attack = min(len(retain_indices), len(nonmember_indices), args.mia_max_samples)
    n_erase = min(len(erase_indices), args.mia_max_samples)
    if n_attack == 0 or n_erase == 0:
        raise ValueError(
            "Cannot build MIA loaders with zero samples: "
            f"n_attack={n_attack}, n_erase={n_erase}, "
            f"num_retained={len(retain_indices)}, num_erased={len(erase_indices)}."
        )
    return (
        make_loader(Subset(train_set, erase_indices[:n_erase]), args.batch_size, shuffle=False),
        make_loader(Subset(train_set, retain_indices[:n_attack]), args.batch_size, shuffle=False),
        make_loader(Subset(test_set, nonmember_indices[:n_attack].tolist()), args.batch_size, shuffle=False),
    )


@torch.no_grad()
def collect_mia_features(vib, dataloader, args, max_samples):
    vib.eval()
    features = []
    for x, y in dataloader:
        x, y = x.to(args.device), y.to(args.device)
        _, logits_y, _, mu, logvar = vib(x, mode="with_reconstruction")
        probs = torch.softmax(logits_y, dim=1)
        true_probs = probs.gather(1, y.view(-1, 1)).squeeze(1)
        losses = F.cross_entropy(logits_y, y, reduction="none")
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1)
        top2 = torch.topk(probs, k=2, dim=1).values
        margin = top2[:, 0] - top2[:, 1]
        kld = mu.pow(2).add(logvar.exp()).mul(-1).add(1).add(logvar).mean(dim=1).mul(-0.5)
        features.append(torch.stack([true_probs, -losses, entropy, margin, -kld], dim=1).cpu())
        if sum(len(batch) for batch in features) >= max_samples:
            break
    if not features:
        raise ValueError("MIA feature collection received an empty dataloader.")
    return torch.cat(features, dim=0)[:max_samples]


class LearnedMIAAttack(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def binary_auc(member_scores, nonmember_scores):
    scores = np.concatenate([member_scores, nonmember_scores])
    labels = np.concatenate([np.ones_like(member_scores), np.zeros_like(nonmember_scores)])
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


@torch.no_grad()
def evaluate_attack_classifier(model, features, labels):
    logits = model(features)
    probs = torch.sigmoid(logits)
    acc = ((probs >= 0.5).float() == labels).float().mean().item()
    return probs.cpu().numpy(), acc


def learned_membership_inference_attack(vib, erase_loader, retain_loader, nonmember_loader, args, name):
    retain_features = collect_mia_features(vib, retain_loader, args, args.mia_max_samples)
    nonmember_features = collect_mia_features(vib, nonmember_loader, args, args.mia_max_samples)
    erase_features = collect_mia_features(vib, erase_loader, args, args.mia_max_samples)

    n_attack = min(len(retain_features), len(nonmember_features))
    retain_features = retain_features[:n_attack]
    nonmember_features = nonmember_features[:n_attack]
    train_features = torch.cat([retain_features, nonmember_features], dim=0)
    train_labels = torch.cat([torch.ones(len(retain_features)), torch.zeros(len(nonmember_features))], dim=0)

    mean = train_features.mean(dim=0, keepdim=True)
    std = train_features.std(dim=0, keepdim=True).clamp_min(1e-6)
    train_features = (train_features - mean) / std
    erase_features = (erase_features - mean) / std

    attack_model = LearnedMIAAttack(train_features.size(1))
    optimizer = torch.optim.Adam(attack_model.parameters(), lr=args.mia_attack_lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    for _ in range(args.mia_attack_epochs):
        logits = attack_model(train_features)
        loss = loss_fn(logits, train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_probs, train_acc = evaluate_attack_classifier(attack_model, train_features, train_labels)
    retain_probs = train_probs[:len(retain_features)]
    nonmember_probs = train_probs[len(retain_features):]
    erase_probs, erase_member_acc = evaluate_attack_classifier(attack_model, erase_features, torch.ones(len(erase_features)))
    auc = binary_auc(retain_probs, nonmember_probs)

    print(f"{name} MIA acc: {train_acc:.4f}, MIA AUC: {auc:.4f}")
    return {"mia_acc": train_acc, "mia_auc": auc, "erased_member_acc": erase_member_acc}


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return x, y, idx


class DeltaCrossCoder(nn.Module):
    def __init__(self, activation_dim, latent_dim, shared_fraction=0.2, shared_k=16, delta_k=8):
        super().__init__()
        shared_dim = int(round(latent_dim * shared_fraction))
        shared_dim = max(1, min(shared_dim, latent_dim - 1))
        self.shared_dim = shared_dim
        self.delta_dim = latent_dim - shared_dim
        self.shared_k = max(1, min(int(shared_k), self.shared_dim))
        self.delta_k = max(1, min(int(delta_k), self.delta_dim))
        self.encoder = nn.Linear(activation_dim, latent_dim)
        self.decoder_before = nn.Linear(latent_dim, activation_dim)
        self.decoder_after = nn.Linear(latent_dim, activation_dim)

    @staticmethod
    def _batch_topk(z, k):
        if k >= z.size(1):
            return z
        values, indices = torch.topk(z, k=k, dim=1)
        sparse = torch.zeros_like(z)
        sparse.scatter_(1, indices, values)
        return sparse

    def apply_dual_topk(self, z):
        z_shared = self._batch_topk(z[:, :self.shared_dim], self.shared_k)
        z_delta = self._batch_topk(z[:, self.shared_dim:], self.delta_k)
        return torch.cat([z_shared, z_delta], dim=1)

    def delta_mask(self, z):
        z_delta_only = torch.zeros_like(z)
        z_delta_only[:, self.shared_dim:] = z[:, self.shared_dim:]
        return z_delta_only

    def forward(self, h_before, h_after):
        z_before = F.relu(self.encoder(h_before))
        z_after = F.relu(self.encoder(h_after))
        z = self.apply_dual_topk(0.5 * (z_before + z_after))
        recon_before = self.decoder_before(z)
        recon_after = self.decoder_after(z)
        z_delta = self.delta_mask(z)
        delta_pred = F.linear(
            z_delta,
            self.decoder_after.weight - self.decoder_before.weight,
            self.decoder_after.bias - self.decoder_before.bias,
        )
        return z, z_delta, recon_before, recon_after, delta_pred


@torch.no_grad()
def collect_crossleak_activations(before_model, after_model, dataloader, args, max_samples=None):
    before_model.eval()
    after_model.eval()
    h_before_list = []
    h_after_list = []
    label_list = []
    index_list = []
    logit_shift_list = []
    collected = 0

    for x, y, idx in dataloader:
        x = x.to(args.device)
        before_params = before_model.encoder(x)
        after_params = after_model.encoder(x)
        dim_z = before_params.shape[1] // 2
        mu_before = before_params[:, :dim_z]
        mu_after = after_params[:, :dim_z]
        logits_before = before_model.approximator(mu_before).reshape((x.size(0), -1))
        logits_after = after_model.approximator(mu_after).reshape((x.size(0), -1))
        input_mode = getattr(args, "crossleak_input_mode", "mu_logits")
        if input_mode == "mu":
            h_before = mu_before
            h_after = mu_after
        elif input_mode == "logits":
            h_before = logits_before
            h_after = logits_after
        elif input_mode == "mu_logits":
            logit_weight = float(getattr(args, "crossleak_logit_input_weight", 1.0))
            h_before = torch.cat([mu_before, logit_weight * logits_before], dim=1)
            h_after = torch.cat([mu_after, logit_weight * logits_after], dim=1)
        else:
            raise ValueError(f"Unknown crossleak_input_mode={input_mode}")
        logit_shift = torch.norm(logits_before - logits_after, dim=1)

        h_before_list.append(h_before.detach().cpu())
        h_after_list.append(h_after.detach().cpu())
        label_list.append(y.detach().cpu())
        index_list.append(idx.detach().cpu())
        logit_shift_list.append(logit_shift.detach().cpu())
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


def train_crosscoder(h_before, h_after, args):
    stacked = torch.cat([h_before, h_after], dim=0)
    mean = stacked.mean(dim=0, keepdim=True)
    std = stacked.std(dim=0, keepdim=True).clamp_min(1e-6)
    h_before = (h_before - mean) / std
    h_after = (h_after - mean) / std

    crosscoder = DeltaCrossCoder(
        h_before.size(1),
        args.crossleak_latent_dim,
        shared_fraction=args.crossleak_delta_shared_fraction,
        shared_k=args.crossleak_delta_shared_k,
        delta_k=args.crossleak_delta_k,
    ).to(args.device)
    optimizer = torch.optim.Adam(crosscoder.parameters(), lr=args.crossleak_lr, weight_decay=1e-5)
    loader = DataLoader(
        TensorDataset(h_before.float(), h_after.float()),
        batch_size=args.crossleak_batch_size,
        shuffle=True,
        num_workers=0,
    )

    for _ in range(args.crossleak_epochs):
        crosscoder.train()
        for hb, ha in loader:
            hb = hb.to(args.device)
            ha = ha.to(args.device)
            z, z_delta, recon_b, recon_a, delta_pred = crosscoder(hb, ha)
            recon_loss = F.mse_loss(recon_b, hb) + F.mse_loss(recon_a, ha)
            delta_loss = F.mse_loss(delta_pred, ha - hb)
            sparse_loss = z.abs().mean()
            delta_sparse_loss = z_delta.abs().mean()
            loss = (
                recon_loss
                + args.crossleak_l1_lambda * sparse_loss
                + args.crossleak_delta_l1_lambda * delta_sparse_loss
                + args.crossleak_delta_loss_weight * delta_loss
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    crosscoder.eval()
    return crosscoder, mean, std


@torch.no_grad()
def score_crossleak_samples(crosscoder, h_before, h_after, logit_shift, mean, std, args):
    hb = ((h_before - mean) / std).float().to(args.device)
    ha = ((h_after - mean) / std).float().to(args.device)
    z_before_raw = F.relu(crosscoder.encoder(hb))
    z_after_raw = F.relu(crosscoder.encoder(ha))
    z = crosscoder.apply_dual_topk(0.5 * (z_before_raw + z_after_raw))
    z_delta = crosscoder.delta_mask(z)
    dec_before = crosscoder.decoder_before.weight.detach().T
    dec_after = crosscoder.decoder_after.weight.detach().T
    rho = torch.norm(dec_before, dim=1) / (torch.norm(dec_before, dim=1) + torch.norm(dec_after, dim=1) + 1e-12)
    exclusivity = torch.abs(rho - 0.5) * 2.0
    delta_decoder_norm = torch.norm(dec_after - dec_before, dim=1)
    latent_shift = torch.abs(z_delta) * delta_decoder_norm.unsqueeze(0)
    class_start = int(getattr(args, "crossleak_logit_input_start", 0))
    class_end = class_start + int(args.num_classes)
    if class_end > crosscoder.decoder_before.weight.size(0):
        class_start = 0
        class_end = int(args.num_classes)
    class_logit_drop_weight = (
        crosscoder.decoder_before.weight.detach()[class_start:class_end]
        - crosscoder.decoder_after.weight.detach()[class_start:class_end]
    )
    class_contribution = torch.abs(z_delta).mean(dim=0).unsqueeze(1) * class_logit_drop_weight.T
    sample_class_contribution = torch.clamp(torch.abs(z_delta) @ class_logit_drop_weight.T, min=0.0)
    weights = exclusivity
    crosscoder_score = (latent_shift * weights.unsqueeze(0)).sum(dim=1)
    logit_shift = logit_shift.float().to(args.device)
    logit_shift = (logit_shift - logit_shift.mean()) / logit_shift.std().clamp_min(1e-6)
    crosscoder_score = (crosscoder_score - crosscoder_score.mean()) / crosscoder_score.std().clamp_min(1e-6)

    return {
        "z_before": z.cpu(),
        "z_after": (z - z_delta).cpu(),
        "latent_shift": latent_shift.cpu(),
        "score": (crosscoder_score + args.crossleak_logit_weight * logit_shift).cpu(),
        "rho": rho.cpu(),
        "exclusivity": exclusivity.cpu(),
        "class_contribution": class_contribution.cpu(),
        "sample_class_contribution": sample_class_contribution.cpu(),
    }


def label_histogram(labels, num_classes=10):
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes)
    total = counts.sum()
    return {int(i): float(counts[i] / total) for i in range(num_classes) if total > 0 and counts[i] > 0}


def top_class_items(class_scores, class_names, top_k):
    return [
        {"label": int(label), "class_name": class_names[int(label)], "score": float(score)}
        for label, score in sorted(class_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
    ]


def _output_layer_delta_scores(before_model, after_model, args):
    """
    ULIA-style gradient-label mapping baseline.

    The paper reconstructs gradient differences from model-parameter changes and
    maps output-layer gradient magnitudes to labels. Here the pre/post global
    model displacement is used as the effective-gradient proxy.
    """
    label_scores = {label: 0.0 for label in range(args.num_classes)}
    layer_count = 0
    before_params = dict(before_model.named_parameters())
    after_params = dict(after_model.named_parameters())
    default_lr = getattr(args, "hessian_unlearn_lr", getattr(args, "unlearning_lr", args.lr))
    effective_lr = max(float(getattr(args, "ulia_effective_lr", default_lr)), 1e-12)
    bias_weight = float(getattr(args, "ulia_bias_weight", 1.0))

    for name, before_param in before_params.items():
        if name not in after_params:
            continue
        after_param = after_params[name]
        if before_param.shape != after_param.shape:
            continue
        if before_param.ndim == 2 and before_param.shape[0] == args.num_classes:
            grad_diff = (before_param.detach() - after_param.detach()).abs() / effective_lr
            per_label = grad_diff.reshape(args.num_classes, -1).mean(dim=1)
        elif before_param.ndim == 1 and before_param.shape[0] == args.num_classes:
            grad_diff = (before_param.detach() - after_param.detach()).abs() / effective_lr
            per_label = bias_weight * grad_diff
        else:
            continue

        for label, score in enumerate(per_label.detach().cpu().tolist()):
            label_scores[int(label)] += float(score)
        layer_count += 1

    if layer_count == 0:
        print("Warning: ULIA baseline found no output-layer parameters.")
        return label_scores
    return {label: score / layer_count for label, score in label_scores.items()}


def ulia_label_inference(before_model, after_model, class_names, args):
    raw_scores = _output_layer_delta_scores(before_model, after_model, args)
    values = np.asarray([raw_scores[label] for label in range(args.num_classes)], dtype=np.float64)
    mean = float(values.mean())
    std = float(values.std())
    if std < 1e-12:
        z_scores = {label: 0.0 for label in range(args.num_classes)}
    else:
        z_scores = {label: float((raw_scores[label] - mean) / std) for label in range(args.num_classes)}

    known_k = max(1, min(int(getattr(args, "unlearning_class_range", 1)), args.num_classes))
    known_pred = top_class_items(raw_scores, class_names, top_k=max(args.crossleak_eval_ks))

    threshold = float(getattr(args, "ulia_z_threshold", 1.0))
    dynamic_labels = [label for label, score in sorted(z_scores.items(), key=lambda item: item[1], reverse=True) if score >= threshold]
    if not dynamic_labels:
        dynamic_labels = [known_pred[0]["label"]] if known_pred else []
    dynamic_pred = [
        {
            "label": int(label),
            "class_name": class_names[int(label)],
            "score": float(z_scores[int(label)]),
        }
        for label in dynamic_labels
    ]

    return {
        "raw_scores": raw_scores,
        "z_scores": z_scores,
        "known_k_pred": known_pred[:known_k],
        "known_rank_pred": known_pred,
        "dynamic_pred": dynamic_pred,
        "threshold": threshold,
    }


@torch.no_grad()
def confidence_drop_probe_scores(before_model, after_model, public_dataset, args):
    """
    Paper-inspired label inference: use high-confidence public probing samples
    for each class and rank classes by original-to-unlearned confidence drop.
    """
    loader = DataLoader(public_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)
    per_class = {label: [] for label in range(args.num_classes)}

    before_model.eval()
    after_model.eval()
    for x, _ in loader:
        x = x.to(args.device)
        _, logits_before, _, _, _ = before_model(x, mode="with_reconstruction")
        _, logits_after, _, _, _ = after_model(x, mode="with_reconstruction")
        probs_before = torch.softmax(logits_before, dim=1)
        probs_after = torch.softmax(logits_after, dim=1)
        pred_before = probs_before.argmax(dim=1)

        for label in range(args.num_classes):
            mask = pred_before == label
            if not torch.any(mask):
                continue
            conf_before = probs_before[mask, label]
            conf_after = probs_after[mask, label]
            drops = conf_before - conf_after
            for cb, drop in zip(conf_before.detach().cpu().tolist(), drops.detach().cpu().tolist()):
                per_class[label].append((float(cb), float(drop)))

    scores = {}
    for label, values in per_class.items():
        if not values:
            continue
        values = sorted(values, key=lambda item: item[0], reverse=True)
        top_values = values[:args.crossleak_probe_samples_per_class]
        scores[label] = float(np.mean([drop for _, drop in top_values]))
    return scores


def normalize_label_scores(label_scores):
    if not label_scores:
        return {}
    values = np.asarray(list(label_scores.values()), dtype=np.float64)
    min_v = float(values.min())
    max_v = float(values.max())
    if max_v - min_v < 1e-12:
        return {int(label): 0.0 for label in label_scores}
    return {int(label): float((score - min_v) / (max_v - min_v)) for label, score in label_scores.items()}


def combine_label_scores(feature_vote_scores, probe_scores, args):
    fv_norm = normalize_label_scores(feature_vote_scores)
    probe_norm = normalize_label_scores(probe_scores)
    labels = set(fv_norm).union(probe_norm)
    alpha = float(args.crossleak_probe_score_weight)
    return {
        int(label): float(fv_norm.get(label, 0.0) + alpha * probe_norm.get(label, 0.0))
        for label in labels
    }


def delta_z_label_scores(scored, labels, class_names, args):
    labels_np = labels.numpy()
    sample_class_contribution = scored["sample_class_contribution"].numpy()
    scores = {}
    for label in range(args.num_classes):
        mask = labels_np == label
        if not np.any(mask):
            continue
        scores[label] = float(np.mean(sample_class_contribution[mask, label]))
    return scores, top_class_items(scores, class_names, top_k=max(args.crossleak_eval_ks))


def true_erased_label_summary(train_set, client_data_info):
    erase_indices = []
    for info in client_data_info.values():
        erase_indices.extend(info["erase_indices"])
    labels = [dataset_label_at(train_set, idx) for idx in erase_indices]
    hist = label_histogram(labels, num_classes=len(getattr(train_set, "classes", list(range(100)))))
    top_labels = sorted(hist.items(), key=lambda item: item[1], reverse=True)
    return {
        "num_erased": len(labels),
        "label_histogram": hist,
        "top_labels": [{"label": int(label), "fraction": float(frac)} for label, frac in top_labels],
    }


def infer_crossleak_features(scored, labels, indices, class_names, args):
    latent_shift = scored["latent_shift"].numpy()
    z_before = np.abs(scored["z_before"].numpy())
    rho = scored["rho"].numpy()
    exclusivity = scored["exclusivity"].numpy()
    class_contribution = scored["class_contribution"].numpy()
    feature_strength = latent_shift.mean(axis=0) * exclusivity
    top_features = np.argsort(-feature_strength)[:args.crossleak_top_features]
    labels_np = labels.numpy()
    indices_np = indices.numpy()
    features = []

    for feature_id in top_features:
        sample_scores = latent_shift[:, feature_id] * max(float(exclusivity[feature_id]), 1e-12)
        top_pos = np.argsort(-sample_scores)[:min(args.crossleak_top_samples_per_feature, len(sample_scores))]
        hist = label_histogram(labels_np[top_pos], num_classes=args.num_classes)
        dominant_label = max(hist, key=hist.get) if hist else None
        class_shift_scores = {}
        class_relative_scores = {}
        for label in range(args.num_classes):
            mask = labels_np == label
            if not np.any(mask):
                continue
            mean_shift = float(np.mean(latent_shift[mask, feature_id]))
            mean_before = float(np.mean(z_before[mask, feature_id]))
            relative_shift = mean_shift / (mean_before + 1e-6)
            class_shift_scores[int(label)] = mean_shift
            class_relative_scores[int(label)] = relative_shift
        ranked_class_scores = sorted(class_relative_scores.items(), key=lambda item: item[1], reverse=True)
        top1_score = float(ranked_class_scores[0][1]) if ranked_class_scores else 0.0
        top2_score = float(ranked_class_scores[1][1]) if len(ranked_class_scores) > 1 else 0.0
        score_sum = float(sum(max(score, 0.0) for _, score in ranked_class_scores))
        class_purity = top1_score / (score_sum + 1e-12) if score_sum > 0 else 0.0
        class_margin = (top1_score - top2_score) / (abs(top1_score) + 1e-12) if top1_score > 0 else 0.0
        decoder_contribution_scores = {
            int(label): float(max(class_contribution[feature_id, label], 0.0))
            for label in range(args.num_classes)
        }
        decoder_contribution_scores = {
            label: score for label, score in decoder_contribution_scores.items() if score > 0
        }
        ranked_decoder_contrib = sorted(
            decoder_contribution_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        contrib_top1 = float(ranked_decoder_contrib[0][1]) if ranked_decoder_contrib else 0.0
        contrib_top2 = float(ranked_decoder_contrib[1][1]) if len(ranked_decoder_contrib) > 1 else 0.0
        contrib_sum = float(sum(score for _, score in ranked_decoder_contrib))
        decoder_contribution_purity = contrib_top1 / (contrib_sum + 1e-12) if contrib_sum > 0 else 0.0
        decoder_contribution_margin = (
            (contrib_top1 - contrib_top2) / (abs(contrib_top1) + 1e-12)
            if contrib_top1 > 0 else 0.0
        )
        features.append({
            "feature_id": int(feature_id),
            "direction": "before_exclusive_deleted" if rho[feature_id] >= 0.5 else "after_exclusive_compensation",
            "decoder_before_ratio": float(rho[feature_id]),
            "exclusivity": float(exclusivity[feature_id]),
            "mean_shift": float(latent_shift[:, feature_id].mean()),
            "feature_strength": float(feature_strength[feature_id]),
            "dominant_label": int(dominant_label) if dominant_label is not None else None,
            "dominant_class_name": class_names[int(dominant_label)] if dominant_label is not None else None,
            "top_label_histogram": hist,
            "class_shift_scores": class_shift_scores,
            "class_relative_scores": class_relative_scores,
            "class_purity": float(class_purity),
            "class_margin": float(class_margin),
            "decoder_contribution_scores": decoder_contribution_scores,
            "decoder_contribution_purity": float(decoder_contribution_purity),
            "decoder_contribution_margin": float(decoder_contribution_margin),
            "top_public_indices": [int(i) for i in indices_np[top_pos].tolist()],
        })
    return features


def feature_vote_label_scores(feature_info, class_names, args):
    label_scores = {label: 0.0 for label in range(args.num_classes)}
    for rank, feature in enumerate(feature_info):
        if (
            getattr(args, "crossleak_feature_vote_only_before", True)
            and feature["direction"] != "before_exclusive_deleted"
        ):
            feature["vote_weight"] = 0.0
            continue

        rank_weight = 1.0 / float(rank + 1)
        rho = float(feature["decoder_before_ratio"])
        direction_confidence = max(
            abs(rho - 0.5) * 2.0,
            float(getattr(args, "crossleak_min_direction_confidence", 0.0)),
        )
        direction_weight = direction_confidence * args.crossleak_before_exclusive_vote_bonus
        purity_weight = max(
            float(feature.get("decoder_contribution_margin", 0.0)),
            float(getattr(args, "crossleak_feature_vote_min_purity_weight", 0.05)),
        )
        feature_weight = (
            rank_weight
            * feature["feature_strength"]
            * direction_weight
            * purity_weight
        )
        feature["vote_weight"] = float(feature_weight)
        if feature_weight <= 0:
            continue
        ranked_class_scores = sorted(
            feature.get("decoder_contribution_scores", {}).items(),
            key=lambda item: item[1],
            reverse=True,
        )
        top_m = int(getattr(args, "crossleak_feature_vote_top_classes", 3))
        top_class_scores = [
            (int(label), max(float(score), 0.0))
            for label, score in ranked_class_scores[:top_m]
        ]
        score_sum = sum(score for _, score in top_class_scores)
        if score_sum <= 0:
            continue
        feature["vote_top_classes"] = [
            {
                "label": int(label),
                "class_name": class_names[int(label)],
                "decoder_contribution": float(score),
            }
            for label, score in top_class_scores
        ]
        for label, score in top_class_scores:
            label_scores[int(label)] += feature_weight * float(score / score_sum)
    label_scores = {label: score for label, score in label_scores.items() if score > 0}
    return label_scores, top_class_items(label_scores, class_names, top_k=max(args.crossleak_eval_ks))


def label_ranking_metrics(label_scores, true_erased, class_names, ks, eval_k):
    true_labels = [item["label"] for item in true_erased["top_labels"][:eval_k]]
    true_set = set(true_labels)
    ranked_labels = [int(label) for label, _ in sorted(label_scores.items(), key=lambda item: item[1], reverse=True)]
    ranks = {label: ranked_labels.index(label) + 1 for label in true_set if label in ranked_labels}
    best_rank = min(ranks.values()) if ranks else None
    metrics = {
        "true_labels": true_labels,
        "best_true_label_rank": int(best_rank) if best_rank is not None else None,
    }
    for k in ks:
        metrics[f"hit@{k}"] = bool(true_set.intersection(set(ranked_labels[:k]))) if true_set else False
    metrics["ranked_top_labels"] = [
        {"label": int(label), "class_name": class_names[int(label)], "score": float(label_scores[label])}
        for label in ranked_labels[:max(ks)]
    ]
    return metrics


def label_set_metrics(predicted_labels, true_erased, args):
    k = max(1, min(int(args.unlearning_class_range), args.num_classes))
    true_labels = [item["label"] for item in true_erased["top_labels"][:k]]
    pred_labels = [item["label"] for item in predicted_labels[:k]]
    true_set = set(true_labels)
    pred_set = set(pred_labels)
    intersection = true_set.intersection(pred_set)
    return {
        "eval_k": k,
        "true_labels": true_labels,
        "predicted_labels": pred_labels,
        "num_correct": len(intersection),
        "overlap": float(len(intersection) / k),
        "precision": float(len(intersection) / len(pred_set)) if pred_set else 0.0,
        "recall": float(len(intersection) / len(true_set)) if true_set else 0.0,
        "exact_match": pred_set == true_set if true_set else False,
    }


def feature_inference_metrics(feature_info, true_labels):
    if not feature_info:
        return {
            "before_exclusive_strength_ratio": 0.0,
            "target_label_feature_coverage": 0.0,
            "top_feature_mean_shift": 0.0,
        }
    total_strength = sum(abs(float(f["feature_strength"])) for f in feature_info)
    before_strength = sum(
        abs(float(f["feature_strength"]))
        for f in feature_info
        if f["direction"] == "before_exclusive_deleted"
    )
    true_set = set(true_labels)
    covered_labels = set()
    for f in feature_info:
        covered_labels.update(
            true_set.intersection(set(int(label) for label in f["top_label_histogram"].keys()))
        )
    return {
        "before_exclusive_strength_ratio": float(before_strength / total_strength) if total_strength > 0 else 0.0,
        "target_label_feature_coverage": float(len(covered_labels) / len(true_set)) if true_set else 0.0,
        "top_feature_mean_shift": float(np.mean([float(f["mean_shift"]) for f in feature_info])),
    }


def erased_sensitive_feature_overlap(
    before_model,
    after_model,
    train_set,
    client_data_info,
    crosscoder,
    mean,
    std,
    public_feature_info,
    class_names,
    args,
):
    erase_indices = []
    for info in client_data_info.values():
        erase_indices.extend(info["erase_indices"])
    if not erase_indices or not public_feature_info:
        return {
            "overlap_metrics": {},
            "erased_top_features": [],
            "public_top_features": [int(f["feature_id"]) for f in public_feature_info],
        }

    erased_dataset = Subset(train_set, erase_indices)
    erased_loader = DataLoader(
        IndexedDataset(erased_dataset),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
    )
    erased_acts = collect_crossleak_activations(
        before_model,
        after_model,
        erased_loader,
        args,
        max_samples=getattr(args, "crossleak_erased_eval_max_samples", None),
    )
    erased_scored = score_crossleak_samples(
        crosscoder,
        erased_acts["before"],
        erased_acts["after"],
        erased_acts["logit_shift"],
        mean,
        std,
        args,
    )

    latent_shift = erased_scored["latent_shift"].numpy()
    exclusivity = erased_scored["exclusivity"].numpy()
    rho = erased_scored["rho"].numpy()
    erased_strength = latent_shift.mean(axis=0) * exclusivity
    erased_ranked = np.argsort(-erased_strength)
    public_ranked = [int(f["feature_id"]) for f in public_feature_info]
    max_k = max(getattr(args, "crossleak_feature_overlap_ks", [1, 3, 5, 10]))
    erased_top_features = []
    for feature_id in erased_ranked[:max_k]:
        erased_top_features.append({
            "feature_id": int(feature_id),
            "direction": "before_exclusive_deleted" if rho[feature_id] >= 0.5 else "after_exclusive_compensation",
            "decoder_before_ratio": float(rho[feature_id]),
            "erased_strength": float(erased_strength[feature_id]),
            "erased_mean_shift": float(latent_shift[:, feature_id].mean()),
        })

    overlap_metrics = {}
    erased_rank_position = {int(feature_id): rank + 1 for rank, feature_id in enumerate(erased_ranked.tolist())}
    for k in getattr(args, "crossleak_feature_overlap_ks", [1, 3, 5, 10]):
        k = min(int(k), len(public_ranked), len(erased_ranked))
        if k <= 0:
            continue
        public_top = set(public_ranked[:k])
        erased_top = set(int(feature_id) for feature_id in erased_ranked[:k])
        intersection = public_top.intersection(erased_top)
        overlap_metrics[f"overlap@{k}"] = float(len(intersection) / k)
        overlap_metrics[f"public_top_mean_erased_rank@{k}"] = float(
            np.mean([erased_rank_position[feature_id] for feature_id in public_ranked[:k]])
        )
        overlap_metrics[f"public_top_mean_erased_strength@{k}"] = float(
            np.mean([erased_strength[feature_id] for feature_id in public_ranked[:k]])
        )

    return {
        "overlap_metrics": overlap_metrics,
        "erased_top_features": erased_top_features,
        "public_top_features": public_ranked,
    }


def compact_label_items(items, top_k=10):
    return [(item["class_name"], int(item["label"]), round(float(item["score"]), 6)) for item in items[:top_k]]


def histogram_visual_concepts(label_histogram, class_names, top_k=5):
    items = sorted(label_histogram.items(), key=lambda item: item[1], reverse=True)[:top_k]
    return [(class_names[int(label)], int(label), round(float(frac), 4)) for label, frac in items]


def safe_name(name):
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(name)).strip("_")


def denormalize_cifar10(images):
    mean = torch.tensor([0.4914, 0.4822, 0.4465], dtype=images.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616], dtype=images.dtype).view(1, 3, 1, 1)
    return (images.cpu() * std + mean).clamp(0.0, 1.0)


def load_clip_concept_model(args):
    if not getattr(args, "crossleak_use_clip_concepts", False):
        return None
    try:
        import open_clip
    except ImportError:
        print("  CLIP concept recovery skipped: open_clip is not installed.")
        return None
    try:
        model, _, _ = open_clip.create_model_and_transforms(
            args.crossleak_clip_model,
            pretrained=args.crossleak_clip_pretrained,
            device=args.device,
        )
        tokenizer = open_clip.get_tokenizer(args.crossleak_clip_model)
    except Exception as exc:
        print(f"  CLIP concept recovery skipped: failed to load CLIP ({exc}).")
        return None
    model.eval()
    return {"model": model, "tokenizer": tokenizer}


@torch.no_grad()
def build_clip_text_features(clip_state, class_names, args):
    prompts = [
        args.crossleak_clip_prompt_template.format(class_name=name.replace("_", " "))
        for name in class_names
    ]
    tokens = clip_state["tokenizer"](prompts).to(args.device)
    text_features = clip_state["model"].encode_text(tokens)
    text_features = F.normalize(text_features.float(), dim=-1)
    return text_features


def clip_preprocess_cifar10(images, args):
    images = denormalize_cifar10(images)
    images = F.interpolate(
        images,
        size=(args.crossleak_clip_image_size, args.crossleak_clip_image_size),
        mode="bicubic",
        align_corners=False,
    )
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=images.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=images.dtype).view(1, 3, 1, 1)
    return (images - mean) / std


@torch.no_grad()
def recover_clip_semantic_concepts(feature_info, public_dataset, class_names, args):
    clip_state = load_clip_concept_model(args)
    if clip_state is None:
        return None
    text_features = build_clip_text_features(clip_state, class_names, args)
    top_k = int(args.crossleak_clip_top_concepts)
    recovered_entries = []

    for feature in feature_info[:args.crossleak_clip_features]:
        images = []
        for idx in feature.get("top_public_indices", [])[:args.crossleak_clip_top_images]:
            x, _ = public_dataset[int(idx)]
            images.append(x)
        if not images:
            feature["clip_concepts"] = []
            continue
        image_tensor = clip_preprocess_cifar10(torch.stack(images), args).to(args.device)
        image_features = clip_state["model"].encode_image(image_tensor)
        image_features = F.normalize(image_features.float(), dim=-1)
        concept_scores = (image_features @ text_features.T).mean(dim=0)
        top_scores, top_labels = torch.topk(concept_scores, k=min(top_k, len(class_names)))
        concepts = [
            {
                "class_name": class_names[int(label)].replace("_", " "),
                "label": int(label),
                "score": float(score),
            }
            for score, label in zip(top_scores.detach().cpu().tolist(), top_labels.detach().cpu().tolist())
        ]
        feature["clip_concepts"] = concepts
        if feature.get("direction") == "before_exclusive_deleted":
            recovered_entries.extend((item["label"], item["score"]) for item in concepts)

    return recovered_entries


def clip_concept_recovery_metrics(recovered_entries, true_labels):
    true_set = set(int(label) for label in true_labels)
    if not true_set:
        return {
            "clip_concept_coverage": 0.0,
            "clip_concept_precision": 0.0,
            "clip_concept_weighted_coverage": 0.0,
        }
    if not recovered_entries:
        return {
            "clip_concept_coverage": 0.0,
            "clip_concept_precision": 0.0,
            "clip_concept_weighted_coverage": 0.0,
        }
    recovered_labels = [int(label) for label, _ in recovered_entries]
    recovered_set = set(recovered_labels)
    correct = [label for label in recovered_labels if label in true_set]
    best_score_by_true_label = {label: 0.0 for label in true_set}
    for label, score in recovered_entries:
        label = int(label)
        if label in best_score_by_true_label:
            best_score_by_true_label[label] = max(best_score_by_true_label[label], float(score))
    max_score = max((float(score) for _, score in recovered_entries), default=0.0)
    weighted_coverage = (
        float(np.mean([score / max_score for score in best_score_by_true_label.values()]))
        if max_score > 0 else 0.0
    )
    return {
        "clip_concept_coverage": float(len(recovered_set.intersection(true_set)) / len(true_set)),
        "clip_concept_precision": float(len(correct) / len(recovered_labels)) if recovered_labels else 0.0,
        "clip_concept_weighted_coverage": weighted_coverage,
    }


def save_top_activating_sample_grids(scored, labels, indices, feature_info, public_dataset, class_names, args, name):
    if not getattr(args, "crossleak_save_top_activating_samples", True):
        return None
    output_dir = os.path.join(args.results_dir, "crossleak_top_activating_samples")
    os.makedirs(output_dir, exist_ok=True)
    base = safe_name(name)
    z_before = scored["z_before"].numpy()
    labels_np = labels.numpy()
    indices_np = indices.numpy()
    saved_paths = []
    top_n = int(args.crossleak_top_activating_samples)

    for feature in feature_info[:args.crossleak_save_top_feature_grids]:
        feature_id = int(feature["feature_id"])
        top_pos = np.argsort(-z_before[:, feature_id])[:min(top_n, len(z_before))]
        images = []
        top_labels = []
        top_acts = []
        for pos in top_pos:
            x, _ = public_dataset[int(indices_np[pos])]
            images.append(x)
            top_labels.append(int(labels_np[pos]))
            top_acts.append(float(z_before[pos, feature_id]))
        if not images:
            continue
        image_tensor = denormalize_cifar10(torch.stack(images))
        grid = make_grid(image_tensor, nrow=min(len(images), args.crossleak_grid_nrow), padding=2)
        label_text = "_".join(class_names[label] for label in top_labels[:min(4, len(top_labels))])
        filename = (
            f"{base}_z{feature_id}_{feature['direction']}_"
            f"rho{feature['decoder_before_ratio']:.3f}_labels_{safe_name(label_text)}.png"
        )
        path = os.path.join(output_dir, filename)
        save_image(grid, path)
        saved_paths.append({
            "feature_id": feature_id,
            "path": path,
            "labels": top_labels,
            "class_names": [class_names[label] for label in top_labels],
            "activations": top_acts,
        })
    return saved_paths


def crossleak_attack(
    before_model,
    after_model,
    public_dataset,
    train_set_for_eval,
    client_data_info,
    args,
    name="model",
    print_class_mean=True,
):
    class_names = getattr(public_dataset, "classes", [str(i) for i in range(args.num_classes)])
    public_loader = DataLoader(
        IndexedDataset(public_dataset),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
    )
    acts = collect_crossleak_activations(before_model, after_model, public_loader, args, max_samples=args.crossleak_max_samples)
    crosscoder, mean, std = train_crosscoder(acts["before"], acts["after"], args)
    scored = score_crossleak_samples(crosscoder, acts["before"], acts["after"], acts["logit_shift"], mean, std, args)

    labels = acts["labels"]
    scores = scored["score"].numpy()
    class_scores = {}
    for label in range(args.num_classes):
        mask = labels.numpy() == label
        if np.any(mask):
            class_scores[label] = float(np.mean(scores[mask]))
    class_mean_pred = top_class_items(class_scores, class_names, top_k=max(args.crossleak_eval_ks))

    features = infer_crossleak_features(scored, labels, acts["indices"], class_names, args)
    feature_vote_scores, feature_vote_pred = feature_vote_label_scores(features, class_names, args)
    delta_z_scores, delta_z_pred = delta_z_label_scores(scored, labels, class_names, args)
    probe_scores = confidence_drop_probe_scores(before_model, after_model, public_dataset, args)
    probe_pred = top_class_items(probe_scores, class_names, top_k=max(args.crossleak_eval_ks))
    hybrid_scores = combine_label_scores(feature_vote_scores, probe_scores, args)
    hybrid_pred = top_class_items(hybrid_scores, class_names, top_k=max(args.crossleak_eval_ks))
    ulia = ulia_label_inference(before_model, after_model, class_names, args)
    true_erased = true_erased_label_summary(train_set_for_eval, client_data_info)
    fv_set = label_set_metrics(feature_vote_pred, true_erased, args)
    delta_z_set = label_set_metrics(delta_z_pred, true_erased, args)
    probe_set = label_set_metrics(probe_pred, true_erased, args)
    hybrid_set = label_set_metrics(hybrid_pred, true_erased, args)
    ulia_known_set = label_set_metrics(ulia["known_k_pred"], true_erased, args)
    ulia_dynamic_set = label_set_metrics(ulia["dynamic_pred"], true_erased, args)
    cm_set = label_set_metrics(class_mean_pred, true_erased, args)
    fv_rank = label_ranking_metrics(feature_vote_scores, true_erased, class_names, args.crossleak_eval_ks, fv_set["eval_k"])
    delta_z_rank = label_ranking_metrics(delta_z_scores, true_erased, class_names, args.crossleak_eval_ks, fv_set["eval_k"])
    probe_rank = label_ranking_metrics(probe_scores, true_erased, class_names, args.crossleak_eval_ks, fv_set["eval_k"])
    hybrid_rank = label_ranking_metrics(hybrid_scores, true_erased, class_names, args.crossleak_eval_ks, fv_set["eval_k"])
    ulia_rank = label_ranking_metrics(ulia["raw_scores"], true_erased, class_names, args.crossleak_eval_ks, fv_set["eval_k"])
    cm_rank = label_ranking_metrics(class_scores, true_erased, class_names, args.crossleak_eval_ks, fv_set["eval_k"])
    feat_metrics = feature_inference_metrics(features, fv_set["true_labels"])
    erased_overlap = erased_sensitive_feature_overlap(
        before_model,
        after_model,
        train_set_for_eval,
        client_data_info,
        crosscoder,
        mean,
        std,
        features,
        class_names,
        args,
    )
    clip_entries = recover_clip_semantic_concepts(features, public_dataset, class_names, args)
    clip_metrics = clip_concept_recovery_metrics(clip_entries, fv_set["true_labels"])
    top_grid_paths = save_top_activating_sample_grids(
        scored,
        labels,
        acts["indices"],
        features,
        public_dataset,
        class_names,
        args,
        name,
    )

    print(f"\nSA-CrossLeak results for {name}:")
    print(f"  true_labels={fv_set['true_labels']} erased_hist={true_erased['label_histogram']}")
    print(f"  feature_vote_pred={compact_label_items(feature_vote_pred)}")
    print(f"  delta_z_pred={compact_label_items(delta_z_pred)}")
    print(f"  confidence_drop_probe_pred={compact_label_items(probe_pred)}")
    print(f"  hybrid_pred={compact_label_items(hybrid_pred)}")
    print(f"  ulia_known_k_pred={compact_label_items(ulia['known_k_pred'])}")
    print(f"  ulia_dynamic_pred={compact_label_items(ulia['dynamic_pred'])}")
    if print_class_mean:
        print(f"  class_mean_pred={compact_label_items(class_mean_pred)}")
    print(
        "  feature_vote_label_infer: "
        + ", ".join([f"hit@{k}={fv_rank[f'hit@{k}']}" for k in args.crossleak_eval_ks])
        + f", precision={fv_set['precision']:.4f}, recall={fv_set['recall']:.4f}, "
        + f"overlap={fv_set['overlap']:.4f}, best_rank={fv_rank['best_true_label_rank']}"
    )
    print(
        "  delta_z_label_infer: "
        + ", ".join([f"hit@{k}={delta_z_rank[f'hit@{k}']}" for k in args.crossleak_eval_ks])
        + f", precision={delta_z_set['precision']:.4f}, recall={delta_z_set['recall']:.4f}, "
        + f"overlap={delta_z_set['overlap']:.4f}, best_rank={delta_z_rank['best_true_label_rank']}"
    )
    print(
        "  confidence_drop_probe_label_infer: "
        + ", ".join([f"hit@{k}={probe_rank[f'hit@{k}']}" for k in args.crossleak_eval_ks])
        + f", precision={probe_set['precision']:.4f}, recall={probe_set['recall']:.4f}, "
        + f"overlap={probe_set['overlap']:.4f}, best_rank={probe_rank['best_true_label_rank']}"
    )
    print(
        "  hybrid_label_infer: "
        + ", ".join([f"hit@{k}={hybrid_rank[f'hit@{k}']}" for k in args.crossleak_eval_ks])
        + f", precision={hybrid_set['precision']:.4f}, recall={hybrid_set['recall']:.4f}, "
        + f"overlap={hybrid_set['overlap']:.4f}, best_rank={hybrid_rank['best_true_label_rank']}"
    )
    print(
        "  ulia_known_k_label_infer: "
        + ", ".join([f"hit@{k}={ulia_rank[f'hit@{k}']}" for k in args.crossleak_eval_ks])
        + f", precision={ulia_known_set['precision']:.4f}, recall={ulia_known_set['recall']:.4f}, "
        + f"overlap={ulia_known_set['overlap']:.4f}, best_rank={ulia_rank['best_true_label_rank']}"
    )
    print(
        "  ulia_dynamic_label_infer: "
        + f"threshold={ulia['threshold']:.4f}, selected={len(ulia['dynamic_pred'])}, "
        + f"precision={ulia_dynamic_set['precision']:.4f}, recall={ulia_dynamic_set['recall']:.4f}, "
        + f"overlap={ulia_dynamic_set['overlap']:.4f}"
    )
    if print_class_mean:
        print(
            "  class_mean_baseline: "
            + ", ".join([f"hit@{k}={cm_rank[f'hit@{k}']}" for k in args.crossleak_eval_ks])
            + f", overlap={cm_set['overlap']:.4f}, best_rank={cm_rank['best_true_label_rank']}"
        )
    print(
        "  feature_infer_summary: "
        f"before_exclusive_strength_ratio={feat_metrics['before_exclusive_strength_ratio']:.4f}, "
        f"target_label_feature_coverage={feat_metrics['target_label_feature_coverage']:.4f}, "
        f"top_feature_mean_shift={feat_metrics['top_feature_mean_shift']:.4f}"
    )
    if clip_entries is not None:
        print(
            "  clip_concept_recovery: "
            f"coverage={clip_metrics['clip_concept_coverage']:.4f}, "
            f"precision={clip_metrics['clip_concept_precision']:.4f}, "
            f"weighted_coverage={clip_metrics['clip_concept_weighted_coverage']:.4f}"
        )
    if erased_overlap["overlap_metrics"]:
        overlap_text = ", ".join(
            f"{key}={value:.4f}"
            for key, value in erased_overlap["overlap_metrics"].items()
        )
        print(f"  erased_sensitive_feature_overlap: {overlap_text}")
        print(f"  erased_top_sensitive_features={erased_overlap['erased_top_features']}")
    print("  top deletion-sensitive features:")
    for feature in features[:args.crossleak_print_top_features]:
        visual_concepts = histogram_visual_concepts(feature["top_label_histogram"], class_names)
        print(
            f"    z{feature['feature_id']} {feature['direction']}, "
            f"rho={feature['decoder_before_ratio']:.3f}, "
            f"shift={feature['mean_shift']:.3f}, "
            f"strength={feature['feature_strength']:.4f}, "
            f"shift_purity={feature['class_purity']:.4f}, "
            f"shift_margin={feature['class_margin']:.4f}, "
            f"decoder_purity={feature['decoder_contribution_purity']:.4f}, "
            f"decoder_margin={feature['decoder_contribution_margin']:.4f}, "
            f"vote_weight={feature.get('vote_weight', 0.0):.6f}, "
            f"dominant={feature['dominant_class_name']}, "
            f"vote_top_classes={feature.get('vote_top_classes', [])}, "
            f"visual_concepts={visual_concepts}, "
            f"clip_concepts={feature.get('clip_concepts', [])}, "
            f"hist={feature['top_label_histogram']}"
        )
    if top_grid_paths:
        print("  saved top activating sample grids:")
        for item in top_grid_paths:
            top_visual_concepts = histogram_visual_concepts(
                label_histogram(item["labels"], num_classes=args.num_classes),
                class_names,
            )
            print(
                f"    z{item['feature_id']}: {item['path']} "
                f"labels={item['class_names']} "
                f"visual_concepts={top_visual_concepts} "
                f"activations={[round(v, 4) for v in item['activations']]}"
            )

    return {
        "feature_vote_label_metrics": fv_set,
        "delta_z_label_metrics": delta_z_set,
        "confidence_drop_probe_label_metrics": probe_set,
        "hybrid_label_metrics": hybrid_set,
        "ulia_known_k_label_metrics": ulia_known_set,
        "ulia_dynamic_label_metrics": ulia_dynamic_set,
        "class_mean_label_metrics": cm_set,
        "feature_vote_ranking_metrics": fv_rank,
        "delta_z_ranking_metrics": delta_z_rank,
        "confidence_drop_probe_ranking_metrics": probe_rank,
        "hybrid_ranking_metrics": hybrid_rank,
        "ulia_ranking_metrics": ulia_rank,
        "ulia_scores": ulia,
        "class_mean_ranking_metrics": cm_rank,
        "feature_inference_metrics": feat_metrics,
        "clip_concept_recovery_metrics": clip_metrics,
        "erased_sensitive_feature_overlap_metrics": erased_overlap["overlap_metrics"],
        "erased_top_sensitive_features": erased_overlap["erased_top_features"],
        "inferred_unlearned_features": features,
        "top_activating_sample_grids": top_grid_paths,
    }


def configure_args():
    args = args_parser()
    args.gpu = 0
    args.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu != -1 else "cpu")
    args.dataset = "CIFAR10"
    args.num_classes = 10
    args.beta = 0.0001
    args.lr = 0.0001
    args.dimZ = 512
    args.batch_size = 200
    args.num_clients = 10
    args.num_unlearn_clients = 1
    args.unlearning_ratio = float(os.environ.get("CIFAR10_hessian_Unlearning_Ratio", 0.01))
    args.unlearn_target_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    args.unlearning_class_range = int(os.environ.get("CIFAR10_hessian_Unlearning_Class_Range", 1))
    args.global_rounds = 1
    args.local_epochs = 5
    args.hessian_damping = 1e-1
    args.hessian_unlearn_lr = 0.001
    args.hessian_retain_lr = 0.0005
    args.hessian_max_update_norm = 0.01
    args.hessian_max_erase_batches = None
    args.hessian_max_retain_batches = None
    args.hessian_hutchinson_samples = 1
    args.seed = 0
    args.crossleak_max_samples = 5000
    args.crossleak_latent_dim = 256
    args.crossleak_batch_size = 256
    args.crossleak_epochs = 200
    args.crossleak_lr = 0.001
    args.crossleak_l1_lambda = 0.001
    args.crossleak_delta_l1_lambda = 0.001
    args.crossleak_delta_loss_weight = 1.0
    args.crossleak_input_mode = "mu_logits"
    args.crossleak_logit_input_weight = 1.0
    args.crossleak_logit_input_start = args.dimZ
    args.crossleak_delta_shared_fraction = 0.2
    args.crossleak_delta_shared_k = 16
    args.crossleak_delta_k = 8
    args.crossleak_logit_weight = 0.1
    args.crossleak_top_labels = 10
    args.crossleak_eval_ks = [1, 3, 5, 10]
    args.crossleak_top_features = 10
    args.crossleak_top_samples_per_feature = 12
    args.crossleak_feature_overlap_ks = [1, 3, 5, 10]
    args.crossleak_erased_eval_max_samples = 1000
    args.crossleak_before_exclusive_vote_bonus = 1.5
    args.crossleak_after_compensation_vote_weight = 0.5
    args.crossleak_min_direction_confidence = 0.25
    args.crossleak_feature_vote_only_before = True
    args.crossleak_feature_vote_top_classes = 3
    args.crossleak_feature_vote_min_purity_weight = 0.05
    args.crossleak_probe_samples_per_class = 20
    args.crossleak_probe_score_weight = 1.0
    args.crossleak_print_top_features = 5
    args.crossleak_save_top_activating_samples = True
    args.crossleak_top_activating_samples = 16
    args.crossleak_save_top_feature_grids = 5
    args.crossleak_grid_nrow = 8
    args.crossleak_use_clip_concepts = True
    args.crossleak_clip_model = "ViT-B-32"
    args.crossleak_clip_pretrained = "/home/wwq/model/vit_base_patch32_clip_224.openai/open_clip_model.safetensors"
    args.crossleak_clip_prompt_template = "a photo of a {class_name}"
    args.crossleak_clip_image_size = 224
    args.crossleak_clip_features = 10
    args.crossleak_clip_top_images = 16
    args.crossleak_clip_top_concepts = 3
    args.ulia_effective_lr = float(os.environ.get("CIFAR10_Hessian_ULIA_Effective_LR", args.hessian_unlearn_lr))
    args.ulia_z_threshold = float(os.environ.get("CIFAR10_Hessian_ULIA_Z_Threshold", 1.0))
    args.ulia_bias_weight = float(os.environ.get("CIFAR10_Hessian_ULIA_Bias_Weight", 1.0))
    args.results_dir = os.path.dirname(os.path.abspath(__file__))
    return args


def load_cifar10_data(args):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_set = CIFAR10("/home/wwq/Data/data/cifar", train=True, transform=train_transform, download=True)
    test_set = CIFAR10("/home/wwq/Data/data/cifar", train=False, transform=test_transform, download=False)
    train_set_no_aug = CIFAR10("/home/wwq/Data/data/cifar", train=True, transform=test_transform, download=False)
    train_loader = make_loader(train_set, args.batch_size, shuffle=True)
    test_loader = make_loader(test_set, args.batch_size, shuffle=False)
    client_subsets = split_dataset_iid(train_set, args.num_clients, seed=args.seed)
    return train_set, train_set_no_aug, test_set, train_loader, test_loader, client_subsets


def load_global_cifar10_model(args):
    ckpt_path = os.path.join(args.results_dir, "global_vib_cifar10_fedavg.pt")
    ckpt = torch.load(ckpt_path, map_location=args.device)
    model = init_vib(args)
    model.load_state_dict(ckpt["model_state"])
    model.to(args.device)
    model.eval()
    return model


def set_random_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collect_rfu_summary_metrics(acc_before, acc_after, mia_before, mia_after, crossleak):
    return {
        "acc_before": float(acc_before),
        "acc_after": float(acc_after),
        "mia_acc_before": float(mia_before["mia_acc"]),
        "mia_acc_after": float(mia_after["mia_acc"]),
        "feature_vote_precision": float(crossleak["feature_vote_label_metrics"]["precision"]),
        "feature_vote_recall": float(crossleak["feature_vote_label_metrics"]["recall"]),
        "feature_vote_overlap": float(crossleak["feature_vote_label_metrics"]["overlap"]),
        "delta_z_precision": float(crossleak["delta_z_label_metrics"]["precision"]),
        "delta_z_recall": float(crossleak["delta_z_label_metrics"]["recall"]),
        "delta_z_overlap": float(crossleak["delta_z_label_metrics"]["overlap"]),
        "confidence_drop_precision": float(crossleak["confidence_drop_probe_label_metrics"]["precision"]),
        "confidence_drop_recall": float(crossleak["confidence_drop_probe_label_metrics"]["recall"]),
        "confidence_drop_overlap": float(crossleak["confidence_drop_probe_label_metrics"]["overlap"]),
        "hybrid_precision": float(crossleak["hybrid_label_metrics"]["precision"]),
        "hybrid_recall": float(crossleak["hybrid_label_metrics"]["recall"]),
        "hybrid_overlap": float(crossleak["hybrid_label_metrics"]["overlap"]),
        "ulia_known_k_precision": float(crossleak["ulia_known_k_label_metrics"]["precision"]),
        "ulia_known_k_recall": float(crossleak["ulia_known_k_label_metrics"]["recall"]),
        "ulia_known_k_overlap": float(crossleak["ulia_known_k_label_metrics"]["overlap"]),
        "ulia_dynamic_precision": float(crossleak["ulia_dynamic_label_metrics"]["precision"]),
        "ulia_dynamic_recall": float(crossleak["ulia_dynamic_label_metrics"]["recall"]),
        "ulia_dynamic_overlap": float(crossleak["ulia_dynamic_label_metrics"]["overlap"]),
        "before_exclusive_strength_ratio": float(crossleak["feature_inference_metrics"]["before_exclusive_strength_ratio"]),
        "target_label_feature_coverage": float(crossleak["feature_inference_metrics"]["target_label_feature_coverage"]),
        "clip_concept_coverage": float(crossleak["clip_concept_recovery_metrics"]["clip_concept_coverage"]),
        "clip_concept_precision": float(crossleak["clip_concept_recovery_metrics"]["clip_concept_precision"]),
        "clip_weighted_coverage": float(crossleak["clip_concept_recovery_metrics"]["clip_concept_weighted_coverage"]),
        "feature_overlap@1": float(crossleak["erased_sensitive_feature_overlap_metrics"].get("overlap@1", 0.0)),
        "feature_overlap@5": float(crossleak["erased_sensitive_feature_overlap_metrics"].get("overlap@5", 0.0)),
        "feature_overlap@10": float(crossleak["erased_sensitive_feature_overlap_metrics"].get("overlap@10", 0.0)),
    }


def print_single_seed_rfu_summary(seed, metrics, mia_before, mia_after):
    print(
        f"Hessian unlearning summary seed={seed}: "
        f"acc_before={metrics['acc_before']:.4f}, acc_after={metrics['acc_after']:.4f}, "
        f"mia_acc_before={metrics['mia_acc_before']:.4f}, mia_auc_before={mia_before['mia_auc']:.4f}, "
        f"mia_acc_after={metrics['mia_acc_after']:.4f}, mia_auc_after={mia_after['mia_auc']:.4f}, "
        f"feature_vote_precision={metrics['feature_vote_precision']:.4f}, "
        f"feature_vote_recall={metrics['feature_vote_recall']:.4f}, "
        f"feature_vote_overlap={metrics['feature_vote_overlap']:.4f}, "
        f"delta_z_precision={metrics['delta_z_precision']:.4f}, "
        f"delta_z_recall={metrics['delta_z_recall']:.4f}, "
        f"delta_z_overlap={metrics['delta_z_overlap']:.4f}, "
        f"confidence_drop_precision={metrics['confidence_drop_precision']:.4f}, "
        f"confidence_drop_recall={metrics['confidence_drop_recall']:.4f}, "
        f"confidence_drop_overlap={metrics['confidence_drop_overlap']:.4f}, "
        f"hybrid_precision={metrics['hybrid_precision']:.4f}, "
        f"hybrid_recall={metrics['hybrid_recall']:.4f}, "
        f"hybrid_overlap={metrics['hybrid_overlap']:.4f}, "
        f"ulia_known_k_precision={metrics['ulia_known_k_precision']:.4f}, "
        f"ulia_known_k_recall={metrics['ulia_known_k_recall']:.4f}, "
        f"ulia_known_k_overlap={metrics['ulia_known_k_overlap']:.4f}, "
        f"ulia_dynamic_precision={metrics['ulia_dynamic_precision']:.4f}, "
        f"ulia_dynamic_recall={metrics['ulia_dynamic_recall']:.4f}, "
        f"ulia_dynamic_overlap={metrics['ulia_dynamic_overlap']:.4f}, "
        f"before_exclusive_strength_ratio={metrics['before_exclusive_strength_ratio']:.4f}, "
        f"target_label_feature_coverage={metrics['target_label_feature_coverage']:.4f}, "
        f"clip_concept_coverage={metrics['clip_concept_coverage']:.4f}, "
        f"clip_concept_precision={metrics['clip_concept_precision']:.4f}, "
        f"clip_weighted_coverage={metrics['clip_weighted_coverage']:.4f}, "
        f"feature_overlap@1={metrics['feature_overlap@1']:.4f}, "
        f"feature_overlap@5={metrics['feature_overlap@5']:.4f}, "
        f"feature_overlap@10={metrics['feature_overlap@10']:.4f}"
    )


def print_multi_seed_rfu_summary(seed_metrics):
    metric_names = [
        "acc_before",
        "acc_after",
        "mia_acc_before",
        "mia_acc_after",
        "feature_vote_precision",
        "feature_vote_recall",
        "feature_vote_overlap",
        "delta_z_precision",
        "delta_z_recall",
        "delta_z_overlap",
        "confidence_drop_precision",
        "confidence_drop_recall",
        "confidence_drop_overlap",
        "hybrid_precision",
        "hybrid_recall",
        "hybrid_overlap",
        "ulia_known_k_precision",
        "ulia_known_k_recall",
        "ulia_known_k_overlap",
        "ulia_dynamic_precision",
        "ulia_dynamic_recall",
        "ulia_dynamic_overlap",
        "before_exclusive_strength_ratio",
        "target_label_feature_coverage",
        "clip_concept_coverage",
        "clip_concept_precision",
        "clip_weighted_coverage",
        "feature_overlap@1",
        "feature_overlap@5",
        "feature_overlap@10",
    ]
    print("\nHessian unlearning summary across seeds:")
    for name in metric_names:
        values = np.array([metrics[name] for metrics in seed_metrics], dtype=np.float64)
        print(f"  {name}: mean={values.mean():.4f}, variance={values.var():.6f}")


def run_rfu_for_seed(seed):
    args = configure_args()
    args.seed = seed
    set_random_seed(args.seed)
    print(f"\n========== Hessian unlearning seed {seed} ==========")
    print("\n".join(f"{k}={v}" for k, v in vars(args).items()))
    print("unlearning target classes:", resolve_unlearning_target_classes(args))

    train_set, train_set_no_aug, test_set, _, test_loader, client_subsets = load_cifar10_data(args)
    global_vib = load_global_cifar10_model(args)
    before_vib = copy.deepcopy(global_vib).eval()

    acc_before = eva_vib(before_vib, test_loader, args, name="before unlearning")

    client_data_info = {}
    new_global_vib = global_vib
    mia_loaders = None
    for rnd in range(1, args.global_rounds + 1):
        new_global_vib, chosen_ids, client_data_info = federated_unlearning_one_round(
            new_global_vib,
            train_set,
            client_subsets,
            args,
            base_seed=args.seed,
            round_idx=rnd,
        )
        eva_vib(new_global_vib, test_loader, args, name=f"global round {rnd}")

        crossleak = crossleak_attack(
            before_model= copy.deepcopy(before_vib),
            after_model= copy.deepcopy(new_global_vib),
            public_dataset=test_set,
            train_set_for_eval=train_set_no_aug,
            client_data_info=client_data_info,
            args=args,
            name=f"seed {seed} global round {rnd}",
        )

    acc_after = eva_vib(new_global_vib, test_loader, args, name="after unlearning")



    if mia_loaders is None:
        mia_loaders = build_learned_mia_loaders(
            train_set_no_aug, test_set, client_data_info, args, seed=args.seed + 2024
        )
    erase_loader, retain_loader, nonmember_loader = mia_loaders
    mia_before = learned_membership_inference_attack(copy.deepcopy(before_vib), erase_loader, retain_loader, nonmember_loader, args, "before unlearning")
    mia_after = learned_membership_inference_attack(copy.deepcopy(new_global_vib), erase_loader, retain_loader, nonmember_loader, args, "after unlearning")



    metrics = collect_rfu_summary_metrics(acc_before, acc_after, mia_before, mia_after, crossleak)
    print_single_seed_rfu_summary(seed, metrics, mia_before, mia_after)
    return metrics


def run_rfu():
    seeds = [0, 1, 2, 3, 4]
    seed_metrics = []
    for seed in seeds:
        seed_metrics.append(run_rfu_for_seed(seed))
    print_multi_seed_rfu_summary(seed_metrics)


if __name__ == "__main__":
    run_rfu()
