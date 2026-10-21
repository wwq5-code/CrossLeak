# CrossLeak

## Deleted-Data Inference in Secure Federated Unlearning

## Overview

This repository is the official implementation of **CrossLeak**, an attack framework for studying privacy leakage from the model updates produced by federated unlearning. CrossLeak compares a model before and after unlearning and evaluates whether the resulting representation changes reveal information about the deleted data, including deleted labels and sensitive features.

The code provides federated learning and unlearning experiments on **MNIST**, **CIFAR-10**, **CIFAR-100**, and **Tiny-ImageNet**. It includes the RFU-based unlearning setting used by CrossLeak, retraining and Hessian-based baselines, backdoor experiments, membership-inference evaluation, feature/label inference, and scripts for reproducing the result tables and figures.

The corresponding paper is under review.

## Prerequisites

The experiments require Python 3 and a CUDA-capable PyTorch environment. The main Python packages used by this repository are:

```ini
python >= 3.10
torch
torchvision
numpy
matplotlib
pandas
pillow
tqdm
```

CrossLeak's optional CLIP-based concept analysis additionally requires `open_clip_torch` and a compatible CLIP checkpoint.

An example installation is:

```bash
pip install torch torchvision numpy matplotlib pandas pillow tqdm open_clip_torch
```

## Repository Structure

```text
CrossLeak/
├── On_MNIST/          # MNIST federated learning and unlearning experiments
├── On_CIFAR10/        # CIFAR-10 CrossLeak experiments and baselines
├── On_CIFAR100/       # CIFAR-100 CrossLeak experiments and baselines
├── On_TinyImageNet/   # Tiny-ImageNet CrossLeak experiments and baselines
├── Exp_results/       # Result logs, parsed tables, analysis scripts, and figures
├── run_all.sh         # RFU/CrossLeak experiment sweeps
├── run_hessian.sh     # Hessian-unlearning baseline sweeps
└── run_retrain.sh     # Retraining baseline sweeps
```

## Data and Path Configuration

MNIST, CIFAR-10, and CIFAR-100 are loaded through `torchvision`. Tiny-ImageNet must be downloaded separately, for example from the [Tiny-ImageNet dataset page on Kaggle](https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet/data).

Before running the experiments, update the dataset paths in the corresponding Python files. The current research code contains machine-specific absolute paths, including:

```text
# CIFAR datasets
/home/wwq/Data/data/cifar

# Tiny-ImageNet
/home/wwq/Data/data/tiny-imagenet-200/tiny-imagenet-200

# Optional CLIP checkpoint
/home/wwq/model/vit_base_patch32_clip_224.openai/open_clip_model.safetensors
```

The Tiny-ImageNet directory should contain the standard `train/` and `val/` folders.

## Running CrossLeak

Run all commands from the repository root.

### 1. Train the federated models

The unlearning programs load a pretrained federated checkpoint. Train the corresponding model first:

```bash
python On_MNIST/IB_FL.py
python On_CIFAR10/IB_FL.py
python On_CIFAR100/IB_FL.py
python On_TinyImageNet/IB_FL.py
```

Make sure that each generated checkpoint has the filename expected by its unlearning script. The main CrossLeak programs expect the following files in their dataset directories:

```text
On_CIFAR10/global_vib_cifar10_fedavg.pt
On_CIFAR100/global_vib_cifar100_fedavg.pt
On_TinyImageNet/global_vib_tinyimagenet_fedavg.pt
```

### 2. Run the RFU/CrossLeak experiments

To run one experiment on each principal image dataset:

```bash
python On_CIFAR10/IB_FL_local_unlearn_rfu.py
python On_CIFAR100/IB_FL_local_unlearn_rfu.py
python On_TinyImageNet/IB_FL_local_unlearn_rfu.py
```

The deleted-class range and unlearning ratio can be controlled through environment variables. For example:

```bash
CIFAR10_RFU_Unlearning_Class_Range=5 \
CIFAR10_RFU_Unlearning_Ratio=0.02 \
python On_CIFAR10/IB_FL_local_unlearn_rfu.py
```

The corresponding prefixes are `CIFAR100_RFU_` and `TinyImageNet_RFU_` for CIFAR-100 and Tiny-ImageNet.

To run the provided RFU/CrossLeak sweeps:

```bash
bash run_all.sh
```

### 3. Run baseline methods

Retraining baseline:

```bash
bash run_retrain.sh
```

Hessian-unlearning baseline:

```bash
bash run_hessian.sh
```

Tiny-ImageNet retraining jobs are provided separately:

```bash
bash run_retrain_tinyImagenet.sh
bash run_retrain_single.sh
```

### 4. Run backdoor experiments

First train a backdoored federated model, then run CrossLeak on the corresponding unlearning update. For example, on CIFAR-10:

```bash
python On_CIFAR10/IB_FL_with_backdoor.py
python On_CIFAR10/IB_FL_local_unlearn_backdoor_rfu.py
```

Equivalent scripts are available under `On_MNIST/`, `On_CIFAR100/`, and `On_TinyImageNet/`.

## Evaluation

The implementation reports utility and privacy/leakage measurements including:

- test accuracy before and after unlearning;
- membership-inference attack (MIA) accuracy;
- deleted-label inference precision, recall, and overlap;
- sensitive-feature overlap at multiple values of *k*;
- target-label feature coverage; and
- optional CLIP concept coverage and precision.

Precomputed logs, parsed CSV tables, and publication-ready figures are included under `Exp_results/`.

To regenerate the main attack-analysis tables and figures, run:

```bash
python Exp_results/On_CIFAR10/CIFAR10_Attacks_analysis.py
python Exp_results/On_CIFAR100/CIFAR100_Attacks_analysis.py
python Exp_results/On_TinyImageNet/TinyImageNet_Attacks_analysis.py
```

Additional CIFAR-10 scripts evaluate different unlearning ratios, client counts, and model architectures:

```bash
python Exp_results/On_CIFAR10/CIFAR10_Attacks_analysis_ratio.py
python Exp_results/On_CIFAR10/CIFAR10_Attacks_analysis_clients.py
python Exp_results/On_CIFAR10/CIFAR10_Attacks_analysis_different_models.py
```

Generated tables are saved under `tables_attacks*/`, while PNG and PDF figures are saved under `figures_attacks*/` in the corresponding dataset directory.

## Notes

- The experiments are GPU-oriented and can be computationally expensive; the shell sweep scripts launch many jobs and may require substantial GPU memory and runtime.
- `run_retrain.sh` runs up to four experiments concurrently. Reduce `MAX_JOBS` if GPU memory is limited.
- Some scripts use slightly different checkpoint names because they correspond to separate experiment variants. Check each script's `ckpt_path` or `save_path` before running it.
- Disable the CLIP-based analysis in the experiment configuration if no local CLIP checkpoint is available.
