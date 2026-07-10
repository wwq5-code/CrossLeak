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
from torchvision.datasets import CIFAR100


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["CIFAR100"], default="CIFAR100")
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


def fedavg(state_dicts):
    averaged = copy.deepcopy(state_dicts[0])
    for key in averaged:
        if torch.is_floating_point(averaged[key]):
            for state_dict in state_dicts[1:]:
                averaged[key] += state_dict[key].to(averaged[key].device)
            averaged[key] /= len(state_dicts)
    return averaged


def prepare_unlearning(erase_loader, remain_loader, model, loss_fn, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()
    remain_iter = iter(remain_loader)
    for _ in range(args.local_epochs):
        for x_e, y_e in erase_loader:
            try:
                x_r, y_r = next(remain_iter)
            except StopIteration:
                remain_iter = iter(remain_loader)
                x_r, y_r = next(remain_iter)

            x_e, y_e = x_e.to(args.device), y_e.to(args.device)
            x_r, y_r = x_r.to(args.device), y_r.to(args.device)
            _, logits_e, _, mu_e, logvar_e = model(x_e, mode="with_reconstruction")
            _, logits_r, _, mu_r, logvar_r = model(x_r, mode="with_reconstruction")

            kld_e = torch.mean(mu_e.pow(2).add_(logvar_e.exp()).mul_(-1).add_(1).add_(logvar_e)).mul_(-0.5)
            kld_r = torch.mean(mu_r.pow(2).add_(logvar_r.exp()).mul_(-1).add_(1).add_(logvar_r)).mul_(-0.5)
            loss_e = loss_fn(logits_e, y_e)
            loss_r = loss_fn(logits_r, y_r)
            loss = 0.5 * (args.beta * kld_e - loss_e) + 0.5 * (args.beta * kld_r + loss_r)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
            optimizer = torch.optim.Adam(local_model.parameters(), lr=args.lr)
            local_model.train()
            for _ in range(args.local_epochs):
                for x, y in loader:
                    x, y = x.to(args.device), y.to(args.device)
                    _, logits_y, _, mu, logvar = local_model(x, mode="with_reconstruction")
                    kld = torch.mean(mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)).mul_(-0.5)
                    loss = args.beta * kld + loss_fn(logits_y, y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
        else:
            erase_indices, remain_indices = split_client_indices(
                client_subset,
                args.unlearning_ratio,
                seed=base_seed * 1000 + round_idx * 100 + cid,
                target_classes=target_classes,
            )
            erase_loader = make_loader(Subset(train_set, erase_indices), args.batch_size, shuffle=True, drop_last=True)
            remain_loader = make_loader(Subset(train_set, remain_indices), args.batch_size, shuffle=True, drop_last=True)
            local_model = prepare_unlearning(erase_loader, remain_loader, local_model, loss_fn, args)
            client_data_info[cid] = {
                "erase_indices": erase_indices,
                "remain_indices": remain_indices,
                "erase_size": len(erase_indices),
                "remain_size": len(remain_indices),
                "target_classes": target_classes,
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


class CrossCoder(nn.Module):
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
        return z_before, z_after, self.decoder_before(z_before), self.decoder_after(z_after)


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
        h_before = before_params[:, :dim_z]
        h_after = after_params[:, :dim_z]
        logits_before = before_model.approximator(h_before).reshape((x.size(0), -1))
        logits_after = after_model.approximator(h_after).reshape((x.size(0), -1))
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

    crosscoder = CrossCoder(h_before.size(1), args.crossleak_latent_dim).to(args.device)
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
            zb, za, recon_b, recon_a = crosscoder(hb, ha)
            recon_loss = F.mse_loss(recon_b, hb) + F.mse_loss(recon_a, ha)
            sparse_loss = zb.abs().mean() + za.abs().mean()
            loss = recon_loss + args.crossleak_l1_lambda * sparse_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    crosscoder.eval()
    return crosscoder, mean, std


@torch.no_grad()
def score_crossleak_samples(crosscoder, h_before, h_after, logit_shift, mean, std, args):
    hb = ((h_before - mean) / std).float().to(args.device)
    ha = ((h_after - mean) / std).float().to(args.device)
    z_before = crosscoder.encode_before(hb)
    z_after = crosscoder.encode_after(ha)
    dec_before = crosscoder.decoder_before.weight.detach().T
    dec_after = crosscoder.decoder_after.weight.detach().T
    rho = torch.norm(dec_before, dim=1) / (torch.norm(dec_before, dim=1) + torch.norm(dec_after, dim=1) + 1e-12)
    exclusivity = torch.abs(rho - 0.5) * 2.0
    latent_shift = torch.abs(z_before - z_after)
    weights = exclusivity
    crosscoder_score = (latent_shift * weights.unsqueeze(0)).sum(dim=1)
    logit_shift = logit_shift.float().to(args.device)
    logit_shift = (logit_shift - logit_shift.mean()) / logit_shift.std().clamp_min(1e-6)
    crosscoder_score = (crosscoder_score - crosscoder_score.mean()) / crosscoder_score.std().clamp_min(1e-6)

    return {
        "z_before": z_before.cpu(),
        "z_after": z_after.cpu(),
        "latent_shift": latent_shift.cpu(),
        "score": (crosscoder_score + args.crossleak_logit_weight * logit_shift).cpu(),
        "rho": rho.cpu(),
        "exclusivity": exclusivity.cpu(),
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
    rho = scored["rho"].numpy()
    exclusivity = scored["exclusivity"].numpy()
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
            "top_public_indices": [int(i) for i in indices_np[top_pos].tolist()],
        })
    return features


def feature_vote_label_scores(feature_info, class_names, args):
    label_scores = {label: 0.0 for label in range(args.num_classes)}
    for rank, feature in enumerate(feature_info):
        rank_weight = 1.0 / float(rank + 1)
        rho = float(feature["decoder_before_ratio"])
        direction_confidence = max(
            abs(rho - 0.5) * 2.0,
            float(getattr(args, "crossleak_min_direction_confidence", 0.0)),
        )
        if feature["direction"] == "before_exclusive_deleted":
            direction_weight = direction_confidence * args.crossleak_before_exclusive_vote_bonus
        else:
            direction_weight = direction_confidence * float(
                getattr(args, "crossleak_after_compensation_vote_weight", 0.0)
            )
        feature_weight = rank_weight * feature["mean_shift"] * feature["exclusivity"] * direction_weight
        feature["vote_weight"] = float(feature_weight)
        if feature_weight <= 0:
            continue
        for label, frac in feature["top_label_histogram"].items():
            label_scores[int(label)] += feature_weight * float(frac)
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
        "mrr": float(1.0 / best_rank) if best_rank is not None else 0.0,
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
    union = true_set.union(pred_set)
    return {
        "eval_k": k,
        "true_labels": true_labels,
        "predicted_labels": pred_labels,
        "num_correct": len(intersection),
        "precision": float(len(intersection) / len(pred_set)) if pred_set else 0.0,
        "recall": float(len(intersection) / len(true_set)) if true_set else 0.0,
        "jaccard": float(len(intersection) / len(union)) if union else 0.0,
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
    covered = [
        f for f in feature_info
        if true_set.intersection(set(int(label) for label in f["top_label_histogram"].keys()))
    ]
    return {
        "before_exclusive_strength_ratio": float(before_strength / total_strength) if total_strength > 0 else 0.0,
        "target_label_feature_coverage": float(len(covered) / len(feature_info)),
        "top_feature_mean_shift": float(np.mean([float(f["mean_shift"]) for f in feature_info])),
    }


def compact_label_items(items, top_k=10):
    return [(item["class_name"], int(item["label"]), round(float(item["score"]), 6)) for item in items[:top_k]]


def safe_name(name):
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(name)).strip("_")


def denormalize_cifar100(images):
    mean = torch.tensor([0.5071, 0.4867, 0.4408], dtype=images.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.2675, 0.2565, 0.2761], dtype=images.dtype).view(1, 3, 1, 1)
    return (images.cpu() * std + mean).clamp(0.0, 1.0)


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
        image_tensor = denormalize_cifar100(torch.stack(images))
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
    probe_scores = confidence_drop_probe_scores(before_model, after_model, public_dataset, args)
    probe_pred = top_class_items(probe_scores, class_names, top_k=max(args.crossleak_eval_ks))
    hybrid_scores = combine_label_scores(feature_vote_scores, probe_scores, args)
    hybrid_pred = top_class_items(hybrid_scores, class_names, top_k=max(args.crossleak_eval_ks))
    true_erased = true_erased_label_summary(train_set_for_eval, client_data_info)
    fv_set = label_set_metrics(feature_vote_pred, true_erased, args)
    probe_set = label_set_metrics(probe_pred, true_erased, args)
    hybrid_set = label_set_metrics(hybrid_pred, true_erased, args)
    cm_set = label_set_metrics(class_mean_pred, true_erased, args)
    fv_rank = label_ranking_metrics(feature_vote_scores, true_erased, class_names, args.crossleak_eval_ks, fv_set["eval_k"])
    probe_rank = label_ranking_metrics(probe_scores, true_erased, class_names, args.crossleak_eval_ks, fv_set["eval_k"])
    hybrid_rank = label_ranking_metrics(hybrid_scores, true_erased, class_names, args.crossleak_eval_ks, fv_set["eval_k"])
    cm_rank = label_ranking_metrics(class_scores, true_erased, class_names, args.crossleak_eval_ks, fv_set["eval_k"])
    feat_metrics = feature_inference_metrics(features, fv_set["true_labels"])
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
    print(f"  confidence_drop_probe_pred={compact_label_items(probe_pred)}")
    print(f"  hybrid_pred={compact_label_items(hybrid_pred)}")
    if print_class_mean:
        print(f"  class_mean_pred={compact_label_items(class_mean_pred)}")
    print(
        "  feature_vote_label_infer: "
        + ", ".join([f"hit@{k}={fv_rank[f'hit@{k}']}" for k in args.crossleak_eval_ks])
        + f", precision={fv_set['precision']:.4f}, recall={fv_set['recall']:.4f}, "
        + f"jaccard={fv_set['jaccard']:.4f}, best_rank={fv_rank['best_true_label_rank']}, "
        + f"mrr={fv_rank['mrr']:.4f}"
    )
    print(
        "  confidence_drop_probe_label_infer: "
        + ", ".join([f"hit@{k}={probe_rank[f'hit@{k}']}" for k in args.crossleak_eval_ks])
        + f", precision={probe_set['precision']:.4f}, recall={probe_set['recall']:.4f}, "
        + f"jaccard={probe_set['jaccard']:.4f}, best_rank={probe_rank['best_true_label_rank']}, "
        + f"mrr={probe_rank['mrr']:.4f}"
    )
    print(
        "  hybrid_label_infer: "
        + ", ".join([f"hit@{k}={hybrid_rank[f'hit@{k}']}" for k in args.crossleak_eval_ks])
        + f", precision={hybrid_set['precision']:.4f}, recall={hybrid_set['recall']:.4f}, "
        + f"jaccard={hybrid_set['jaccard']:.4f}, best_rank={hybrid_rank['best_true_label_rank']}, "
        + f"mrr={hybrid_rank['mrr']:.4f}"
    )
    if print_class_mean:
        print(
            "  class_mean_baseline: "
            + ", ".join([f"hit@{k}={cm_rank[f'hit@{k}']}" for k in args.crossleak_eval_ks])
            + f", jaccard={cm_set['jaccard']:.4f}, best_rank={cm_rank['best_true_label_rank']}, "
            + f"mrr={cm_rank['mrr']:.4f}"
        )
    print(
        "  feature_infer_summary: "
        f"before_exclusive_strength_ratio={feat_metrics['before_exclusive_strength_ratio']:.4f}, "
        f"target_label_feature_coverage={feat_metrics['target_label_feature_coverage']:.4f}, "
        f"top_feature_mean_shift={feat_metrics['top_feature_mean_shift']:.4f}"
    )
    print("  top deletion-sensitive features:")
    for feature in features[:args.crossleak_print_top_features]:
        print(
            f"    z{feature['feature_id']} {feature['direction']}, "
            f"rho={feature['decoder_before_ratio']:.3f}, "
            f"shift={feature['mean_shift']:.3f}, "
            f"strength={feature['feature_strength']:.4f}, "
            f"vote_weight={feature.get('vote_weight', 0.0):.6f}, "
            f"dominant={feature['dominant_class_name']}, "
            f"hist={feature['top_label_histogram']}"
        )
    if top_grid_paths:
        print("  saved top activating sample grids:")
        for item in top_grid_paths:
            print(
                f"    z{item['feature_id']}: {item['path']} "
                f"labels={item['class_names']} "
                f"activations={[round(v, 4) for v in item['activations']]}"
            )

    return {
        "feature_vote_label_metrics": fv_set,
        "confidence_drop_probe_label_metrics": probe_set,
        "hybrid_label_metrics": hybrid_set,
        "class_mean_label_metrics": cm_set,
        "feature_vote_ranking_metrics": fv_rank,
        "confidence_drop_probe_ranking_metrics": probe_rank,
        "hybrid_ranking_metrics": hybrid_rank,
        "class_mean_ranking_metrics": cm_rank,
        "feature_inference_metrics": feat_metrics,
        "inferred_unlearned_features": features,
        "top_activating_sample_grids": top_grid_paths,
    }


def configure_args():
    args = args_parser()
    args.gpu = 0
    args.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu != -1 else "cpu")
    args.dataset = "CIFAR100"
    args.num_classes = 100
    args.beta = 0.0001
    args.lr = 0.0001
    args.dimZ = 512
    args.batch_size = 200
    args.num_clients = 10
    args.num_unlearn_clients = 1
    args.unlearning_ratio = 0.02
    args.unlearn_target_classes = [0, 1, 2, 3, 4]
    args.unlearning_class_range = 1
    args.global_rounds = 2
    args.local_epochs = 5
    args.seed = 0
    args.crossleak_max_samples = 5000
    args.crossleak_latent_dim = 256
    args.crossleak_batch_size = 256
    args.crossleak_epochs = 200
    args.crossleak_lr = 0.001
    args.crossleak_l1_lambda = 0.001
    args.crossleak_logit_weight = 0.1
    args.crossleak_top_labels = 10
    args.crossleak_eval_ks = [1, 3, 5, 10]
    args.crossleak_top_features = 10
    args.crossleak_top_samples_per_feature = 12
    args.crossleak_before_exclusive_vote_bonus = 1.5
    args.crossleak_after_compensation_vote_weight = 0.5
    args.crossleak_min_direction_confidence = 0.25
    args.crossleak_probe_samples_per_class = 20
    args.crossleak_probe_score_weight = 1.0
    args.crossleak_print_top_features = 5
    args.crossleak_save_top_activating_samples = True
    args.crossleak_top_activating_samples = 16
    args.crossleak_save_top_feature_grids = 5
    args.crossleak_grid_nrow = 8
    args.results_dir = os.path.dirname(os.path.abspath(__file__))
    return args


def load_cifar100_data(args):
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
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
    train_set = CIFAR100("/home/wwq/Data/data/cifar", train=True, transform=train_transform, download=True)
    test_set = CIFAR100("/home/wwq/Data/data/cifar", train=False, transform=test_transform, download=False)
    train_set_no_aug = CIFAR100("/home/wwq/Data/data/cifar", train=True, transform=test_transform, download=False)
    train_loader = make_loader(train_set, args.batch_size, shuffle=True)
    test_loader = make_loader(test_set, args.batch_size, shuffle=False)
    client_subsets = split_dataset_iid(train_set, args.num_clients, seed=args.seed)
    return train_set, train_set_no_aug, test_set, train_loader, test_loader, client_subsets


def load_global_cifar100_model(args):
    ckpt_path = os.path.join(args.results_dir, "global_vib_cifar100_fedavg.pt")
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


def run_rfu():
    args = configure_args()
    set_random_seed(args.seed)
    print("\n".join(f"{k}={v}" for k, v in vars(args).items()))
    print("unlearning target classes:", resolve_unlearning_target_classes(args))

    train_set, train_set_no_aug, test_set, _, test_loader, client_subsets = load_cifar100_data(args)
    global_vib = load_global_cifar100_model(args)
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
            name=f"global round {rnd}",
        )

    acc_after = eva_vib(new_global_vib, test_loader, args, name="after unlearning")



    if mia_loaders is None:
        mia_loaders = build_learned_mia_loaders(
            train_set_no_aug, test_set, client_data_info, args, seed=args.seed + 2024
        )
    erase_loader, retain_loader, nonmember_loader = mia_loaders
    mia_before = learned_membership_inference_attack(copy.deepcopy(before_vib), erase_loader, retain_loader, nonmember_loader, args, "before unlearning")
    mia_after = learned_membership_inference_attack(copy.deepcopy(new_global_vib), erase_loader, retain_loader, nonmember_loader, args, "after unlearning")



    print(
        "RFU summary: "
        f"acc_before={acc_before:.4f}, acc_after={acc_after:.4f}, "
        f"mia_acc_before={mia_before['mia_acc']:.4f}, mia_auc_before={mia_before['mia_auc']:.4f}, "
        f"mia_acc_after={mia_after['mia_acc']:.4f}, mia_auc_after={mia_after['mia_auc']:.4f}, "
        f"cl_jaccard={crossleak['feature_vote_label_metrics']['jaccard']:.4f}, "
        f"cl_mrr={crossleak['feature_vote_ranking_metrics']['mrr']:.4f}, "
        f"hybrid_jaccard={crossleak['hybrid_label_metrics']['jaccard']:.4f}, "
        f"hybrid_mrr={crossleak['hybrid_ranking_metrics']['mrr']:.4f}, "
        f"before_exclusive_strength_ratio={crossleak['feature_inference_metrics']['before_exclusive_strength_ratio']:.4f}, "
        f"target_label_feature_coverage={crossleak['feature_inference_metrics']['target_label_feature_coverage']:.4f}"
    )


if __name__ == "__main__":
    run_rfu()
