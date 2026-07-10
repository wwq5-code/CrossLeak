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
from torch.utils.data import DataLoader, Dataset, Subset
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


def prepare_unlearning(erase_loader, remain_loader, model, loss_fn, args, num_steps):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()
    erase_iter = cycle_loader(erase_loader)
    remain_iter = cycle_loader(remain_loader)
    for _ in range(num_steps):
        x_e, y_e = next(erase_iter)
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
            normal_steps = len(make_loader(client_subset, args.batch_size, shuffle=True, drop_last=True)) * args.local_epochs
            erase_indices, remain_indices = split_client_indices(
                client_subset,
                args.unlearning_ratio,
                seed=base_seed * 1000 + round_idx * 100 + cid,
                target_classes=target_classes,
            )
            erase_loader = make_loader(Subset(train_set, erase_indices), args.batch_size, shuffle=True, drop_last=False)
            remain_loader = make_loader(Subset(train_set, remain_indices), args.batch_size, shuffle=True, drop_last=True)
            local_model = prepare_unlearning(erase_loader, remain_loader, local_model, loss_fn, args, normal_steps)
            client_data_info[cid] = {
                "erase_indices": erase_indices,
                "remain_indices": remain_indices,
                "erase_size": len(erase_indices),
                "remain_size": len(remain_indices),
                "unlearning_steps": normal_steps,
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
def class_level_shift_scores(before_model, after_model, public_dataset, args):
    loader = DataLoader(public_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)
    num_classes = int(args.num_classes)
    counts = torch.zeros(num_classes, dtype=torch.float64)
    confidence_drop_sum = torch.zeros(num_classes, dtype=torch.float64)
    logit_drop_sum = torch.zeros(num_classes, dtype=torch.float64)
    loss_increase_sum = torch.zeros(num_classes, dtype=torch.float64)
    mu_before_sum = None
    mu_after_sum = None
    collected = 0

    before_model.eval()
    after_model.eval()
    for x, y in loader:
        x = x.to(args.device)
        y = y.to(args.device)
        if y.ndim == 2:
            y = y.argmax(dim=1)

        before_params = before_model.encoder(x)
        after_params = after_model.encoder(x)
        dim_z = before_params.shape[1] // 2
        mu_before = before_params[:, :dim_z]
        mu_after = after_params[:, :dim_z]
        logits_before = before_model.approximator(mu_before).reshape((x.size(0), -1))
        logits_after = after_model.approximator(mu_after).reshape((x.size(0), -1))

        probs_before = torch.softmax(logits_before, dim=1)
        probs_after = torch.softmax(logits_after, dim=1)
        row_idx = torch.arange(x.size(0), device=args.device)
        confidence_drop = probs_before[row_idx, y] - probs_after[row_idx, y]
        logit_drop = logits_before[row_idx, y] - logits_after[row_idx, y]
        loss_increase = (
            F.cross_entropy(logits_after, y, reduction="none")
            - F.cross_entropy(logits_before, y, reduction="none")
        )

        if mu_before_sum is None:
            mu_before_sum = torch.zeros(num_classes, dim_z, dtype=torch.float64)
            mu_after_sum = torch.zeros(num_classes, dim_z, dtype=torch.float64)

        y_cpu = y.detach().cpu()
        for label in torch.unique(y_cpu).tolist():
            label = int(label)
            mask = y_cpu == label
            counts[label] += int(mask.sum().item())
            confidence_drop_sum[label] += confidence_drop.detach().cpu()[mask].double().sum()
            logit_drop_sum[label] += logit_drop.detach().cpu()[mask].double().sum()
            loss_increase_sum[label] += loss_increase.detach().cpu()[mask].double().sum()
            mu_before_sum[label] += mu_before.detach().cpu()[mask].double().sum(dim=0)
            mu_after_sum[label] += mu_after.detach().cpu()[mask].double().sum(dim=0)

        collected += x.size(0)
        if getattr(args, "crossleak_max_samples", None) is not None and collected >= args.crossleak_max_samples:
            break

    valid = counts > 0
    confidence_drop = torch.zeros(num_classes, dtype=torch.float64)
    logit_drop = torch.zeros(num_classes, dtype=torch.float64)
    loss_increase = torch.zeros(num_classes, dtype=torch.float64)
    prototype_shift = torch.zeros(num_classes, dtype=torch.float64)
    confidence_drop[valid] = confidence_drop_sum[valid] / counts[valid]
    logit_drop[valid] = logit_drop_sum[valid] / counts[valid]
    loss_increase[valid] = loss_increase_sum[valid] / counts[valid]
    proto_before = mu_before_sum[valid] / counts[valid].unsqueeze(1)
    proto_after = mu_after_sum[valid] / counts[valid].unsqueeze(1)
    prototype_shift[valid] = torch.norm(proto_before - proto_after, dim=1)

    def to_score_dict(values):
        return {int(i): float(values[i].item()) for i in torch.where(valid)[0].tolist()}

    return {
        "confidence_drop": to_score_dict(confidence_drop),
        "logit_drop": to_score_dict(logit_drop),
        "prototype_shift": to_score_dict(prototype_shift),
        "loss_increase": to_score_dict(loss_increase),
        "counts": {int(i): int(counts[i].item()) for i in torch.where(valid)[0].tolist()},
    }


def normalize_label_scores(label_scores):
    if not label_scores:
        return {}
    values = np.asarray(list(label_scores.values()), dtype=np.float64)
    min_v = float(values.min())
    max_v = float(values.max())
    if max_v - min_v < 1e-12:
        return {int(label): 0.0 for label in label_scores}
    return {int(label): float((score - min_v) / (max_v - min_v)) for label, score in label_scores.items()}


def combine_direct_label_scores(score_parts, args):
    weights = {
        "confidence_drop": float(getattr(args, "crossleak_confidence_drop_weight", 1.0)),
        "logit_drop": float(getattr(args, "crossleak_logit_drop_weight", 1.0)),
        "prototype_shift": float(getattr(args, "crossleak_prototype_shift_weight", 1.0)),
        "loss_increase": float(getattr(args, "crossleak_loss_increase_weight", 1.0)),
    }
    labels = set()
    for scores in score_parts.values():
        if isinstance(scores, dict):
            labels.update(scores.keys())

    combined = {int(label): 0.0 for label in labels}
    for name, weight in weights.items():
        normalized = normalize_label_scores(score_parts.get(name, {}))
        for label in labels:
            combined[int(label)] += weight * float(normalized.get(label, 0.0))
    return combined


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


def compact_label_items(items, top_k=10):
    return [(item["class_name"], int(item["label"]), round(float(item["score"]), 6)) for item in items[:top_k]]


def print_direct_metric_line(name, ranking_metrics, set_metrics, args):
    print(
        f"  {name}: "
        + ", ".join([f"hit@{k}={ranking_metrics[f'hit@{k}']}" for k in args.crossleak_eval_ks])
        + f", precision={set_metrics['precision']:.4f}, recall={set_metrics['recall']:.4f}"
    )


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
    true_erased = true_erased_label_summary(train_set_for_eval, client_data_info)
    score_parts = class_level_shift_scores(before_model, after_model, public_dataset, args)
    direct_scores = combine_direct_label_scores(score_parts, args)

    top_k = max(args.crossleak_eval_ks)
    direct_pred = top_class_items(direct_scores, class_names, top_k=top_k)
    confidence_pred = top_class_items(score_parts["confidence_drop"], class_names, top_k=top_k)
    logit_pred = top_class_items(score_parts["logit_drop"], class_names, top_k=top_k)
    prototype_pred = top_class_items(score_parts["prototype_shift"], class_names, top_k=top_k)
    loss_pred = top_class_items(score_parts["loss_increase"], class_names, top_k=top_k)

    direct_set = label_set_metrics(direct_pred, true_erased, args)
    confidence_set = label_set_metrics(confidence_pred, true_erased, args)
    logit_set = label_set_metrics(logit_pred, true_erased, args)
    prototype_set = label_set_metrics(prototype_pred, true_erased, args)
    loss_set = label_set_metrics(loss_pred, true_erased, args)

    direct_rank = label_ranking_metrics(direct_scores, true_erased, class_names, args.crossleak_eval_ks, direct_set["eval_k"])
    confidence_rank = label_ranking_metrics(score_parts["confidence_drop"], true_erased, class_names, args.crossleak_eval_ks, direct_set["eval_k"])
    logit_rank = label_ranking_metrics(score_parts["logit_drop"], true_erased, class_names, args.crossleak_eval_ks, direct_set["eval_k"])
    prototype_rank = label_ranking_metrics(score_parts["prototype_shift"], true_erased, class_names, args.crossleak_eval_ks, direct_set["eval_k"])
    loss_rank = label_ranking_metrics(score_parts["loss_increase"], true_erased, class_names, args.crossleak_eval_ks, direct_set["eval_k"])

    print(f"\nSA-CrossLeak results for {name}:")
    print(f"  true_labels={direct_set['true_labels']} erased_hist={true_erased['label_histogram']}")
    print(f"  direct_hybrid_pred={compact_label_items(direct_pred)}")
    print(f"  confidence_drop_pred={compact_label_items(confidence_pred)}")
    print(f"  logit_drop_pred={compact_label_items(logit_pred)}")
    print(f"  prototype_shift_pred={compact_label_items(prototype_pred)}")
    print(f"  loss_increase_pred={compact_label_items(loss_pred)}")
    print_direct_metric_line("direct_hybrid_label_infer", direct_rank, direct_set, args)
    print_direct_metric_line("confidence_drop_label_infer", confidence_rank, confidence_set, args)
    print_direct_metric_line("logit_drop_label_infer", logit_rank, logit_set, args)
    print_direct_metric_line("prototype_shift_label_infer", prototype_rank, prototype_set, args)
    print_direct_metric_line("loss_increase_label_infer", loss_rank, loss_set, args)

    return {
        "direct_label_metrics": direct_set,
        "confidence_drop_label_metrics": confidence_set,
        "logit_drop_label_metrics": logit_set,
        "prototype_shift_label_metrics": prototype_set,
        "loss_increase_label_metrics": loss_set,
        "direct_ranking_metrics": direct_rank,
        "confidence_drop_ranking_metrics": confidence_rank,
        "logit_drop_ranking_metrics": logit_rank,
        "prototype_shift_ranking_metrics": prototype_rank,
        "loss_increase_ranking_metrics": loss_rank,
        "direct_scores": direct_scores,
        "score_parts": score_parts,
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
    args.unlearn_target_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    args.unlearning_class_range = 10
    args.global_rounds = 1
    args.local_epochs = 5
    args.seed = 0
    args.crossleak_max_samples = 5000
    args.crossleak_eval_ks = [1, 3, 5, 10]
    args.crossleak_confidence_drop_weight = 1.0
    args.crossleak_logit_drop_weight = 1.0
    args.crossleak_prototype_shift_weight = 1.0
    args.crossleak_loss_increase_weight = 1.0
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
        f"direct_precision={crossleak['direct_label_metrics']['precision']:.4f}, "
        f"direct_recall={crossleak['direct_label_metrics']['recall']:.4f}, "
        f"direct_jaccard={crossleak['direct_label_metrics']['jaccard']:.4f}, "
        f"direct_mrr={crossleak['direct_ranking_metrics']['mrr']:.4f}"
    )


if __name__ == "__main__":
    run_rfu()
