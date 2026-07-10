import csv
import math
import os
import re
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
RESULTS_FILE = BASE_DIR / "results_clients.txt"
FIG_DIR = BASE_DIR / "figures_attacks_clients"
TABLE_DIR = BASE_DIR / "tables_attacks_clients"
OUTPUT_PREFIX = "CIFAR10_Attacks_clients"
TABLE_NAME = "parsed_cifar10_attack_clients_results.csv"

os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".mplconfig"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt


SECTION_RE = re.compile(r"^\s*(?P<section>.+?summary across seeds):\s*$")
CLIENTS_RE = re.compile(r"^\s*Clients:\s*(?P<clients>\d+)\s*$")
METRIC_RE = re.compile(
    r"^\s*(?P<metric>[A-Za-z0-9_@]+):\s*"
    r"mean=(?P<mean>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?),\s*"
    r"variance=(?P<variance>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?)\s*$"
)


PALETTE = {
    "feature_vote_overlap": "#797BB7",
    "delta_z_overlap": "#86BBD8",
    "confidence_drop_overlap": "#E07A5F",
    "ulia_known_k_overlap": "#59A14F",
    "ulia_dynamic_overlap": "#E1C855",
    "before_exclusive_strength_ratio": "#797BB7",
    "target_label_feature_coverage": "#86BBD8",
    "clip_concept_coverage": "#E07A5F",
    "clip_weighted_coverage": "#59A14F",
}

LINE_STYLES = {
    "feature_vote_overlap": ("o", "-"),
    "delta_z_overlap": ("s", "--"),
    "confidence_drop_overlap": ("^", "-."),
    "ulia_known_k_overlap": ("D", ":"),
    "ulia_dynamic_overlap": ("v", "-"),
    "before_exclusive_strength_ratio": ("o", "-"),
    "target_label_feature_coverage": ("s", "--"),
    "clip_concept_coverage": ("^", "-."),
    "clip_weighted_coverage": ("D", ":"),
}

EXCLUDED_METRIC_PREFIXES = ("hybrid",)


def method_from_section(section):
    normalized = section.strip().lower()
    if normalized.startswith("rfu"):
        return "RFU"
    if normalized.startswith("hessian unlearning"):
        return "Hessian"
    if normalized.startswith("fl retraining"):
        return "Retraining"
    return None


def parse_results(results_file=RESULTS_FILE):
    rows = []
    current_method = None
    current_row = None

    with results_file.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            section_match = SECTION_RE.match(line)
            if section_match:
                if current_row is not None:
                    rows.append(current_row)
                    current_row = None
                current_method = method_from_section(section_match.group("section"))
                continue

            clients_match = CLIENTS_RE.match(line)
            if clients_match and current_method is not None:
                if current_row is not None:
                    rows.append(current_row)
                current_row = {
                    "method": current_method,
                    "clients": int(clients_match.group("clients")),
                }
                continue

            metric_match = METRIC_RE.match(line)
            if metric_match and current_row is not None:
                metric = metric_match.group("metric")
                mean = float(metric_match.group("mean"))
                variance = float(metric_match.group("variance"))
                current_row[metric] = mean
                current_row[f"{metric}_variance"] = variance
                current_row[f"{metric}_std"] = math.sqrt(max(variance, 0.0))

    if current_row is not None:
        rows.append(current_row)

    if not rows:
        raise RuntimeError(f"No client summary rows parsed from {results_file}")
    return sorted(rows, key=lambda row: (row["method"], row["clients"]))


def configure_plot_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "axes.labelsize": 22,
        "axes.titlesize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 12,
        "axes.linewidth": 1.6,
        "axes.edgecolor": "#222222",
        "grid.color": "#B8B8B8",
        "grid.alpha": 0.7,
        "grid.linestyle": "--",
        "grid.linewidth": 0.9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def polish_axes(ax):
    ax.grid(True, which="major", axis="both")
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(1.6)


def style_legend(legend, alpha=0.78):
    if legend is None:
        return
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_alpha(alpha)
    frame.set_edgecolor("#666666")
    frame.set_linewidth(0.8)


def client_ticks_from_rows(rows):
    return sorted({row["clients"] for row in rows})


def metric_values(rows, metric, scale=1.0):
    available_rows = [row for row in rows if metric in row]
    x = [row["clients"] for row in available_rows]
    y = [row[metric] * scale for row in available_rows]
    std = [row.get(f"{metric}_std", 0.0) * scale for row in available_rows]
    return x, y, std


def plot_with_band(ax, x, y, std, label, color, marker, linestyle="-", linewidth=3.0):
    if not x:
        return
    lower = [max(0.0, value - err) for value, err in zip(y, std)]
    upper = [min(100.0, value + err) for value, err in zip(y, std)]
    ax.plot(
        x,
        y,
        color=color,
        linestyle=linestyle,
        marker=marker,
        linewidth=linewidth,
        markersize=8,
        markeredgewidth=1.5,
        label=label,
    )
    ax.fill_between(x, lower, upper, color=color, alpha=0.16, linewidth=0)


def save_figure(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.pdf", format="pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_tables(rows):
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    metric_names = sorted({
        key
        for row in rows
        for key in row
        if key not in {"method", "clients"} and not key.endswith("_std") and not key.endswith("_variance")
        and not key.startswith(EXCLUDED_METRIC_PREFIXES)
    })
    fieldnames = ["method", "clients"]
    for metric in metric_names:
        fieldnames.extend([metric, f"{metric}_std", f"{metric}_variance"])

    with (TABLE_DIR / TABLE_NAME).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def plot_label_inference_overlap(rows, output_prefix=OUTPUT_PREFIX):
    metrics = [
        ("feature_vote_overlap", "CrossLeak Feature Vote"),
        ("delta_z_overlap", "CrossLeak"),
        ("confidence_drop_overlap", "Confidence Drop"),
        ("ulia_known_k_overlap", "ULIA known-k"),
        ("ulia_dynamic_overlap", "ULIA dynamic"),
    ]
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    for metric, label in metrics:
        marker, linestyle = LINE_STYLES[metric]
        x, y, std = metric_values(rows, metric, scale=100.0)
        plot_with_band(ax, x, y, std, label, PALETTE[metric], marker, linestyle)

    ax.set_xlabel("Number of Unlearned Clients")
    ax.set_ylabel("Label Infer. Overlap (%)")
    ax.set_xticks(client_ticks_from_rows(rows))
    ax.set_ylim(0, 105)
    style_legend(ax.legend(loc="best", fancybox=True), alpha=0.78)
    polish_axes(ax)
    save_figure(fig, f"{output_prefix}_label_inference_overlap")


def plot_feature_inference_performance(rows, output_prefix=OUTPUT_PREFIX):
    metrics = [
        ("before_exclusive_strength_ratio", "Before-exclusive Strength"),
        ("target_label_feature_coverage", "Target Feature Coverage"),
        ("clip_concept_coverage", "CLIP Concept Coverage"),
        ("clip_weighted_coverage", "CLIP Weighted Coverage"),
    ]
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    for metric, label in metrics:
        marker, linestyle = LINE_STYLES[metric]
        x, y, std = metric_values(rows, metric, scale=100.0)
        plot_with_band(ax, x, y, std, label, PALETTE[metric], marker, linestyle)

    ax.set_xlabel("Number of Unlearned Clients")
    ax.set_ylabel("Feature Inference Metric (%)")
    ax.set_xticks(client_ticks_from_rows(rows))
    ax.set_ylim(0, 105)
    style_legend(ax.legend(loc="best", fancybox=True), alpha=0.78)
    polish_axes(ax)
    save_figure(fig, f"{output_prefix}_feature_inference_performance")


def print_summary(rows):
    print("CIFAR-10 client-count attack analysis parsed from results_clients.txt")
    print(f"clients={client_ticks_from_rows(rows)}")
    for metric in [
        "confidence_drop_overlap",
        "delta_z_overlap",
        "feature_vote_overlap",
        "target_label_feature_coverage",
        "clip_concept_coverage",
    ]:
        values = [row[metric] for row in rows if metric in row]
        if values:
            print(f"{metric}: avg={sum(values) / len(values):.4f}, max={max(values):.4f}")
    print(f"Figures will be saved to: {FIG_DIR}")
    print(f"Tables will be saved to: {TABLE_DIR}")


def main():
    configure_plot_style()
    rows = parse_results(RESULTS_FILE)
    save_tables(rows)
    plot_label_inference_overlap(rows)
    plot_feature_inference_performance(rows)
    print_summary(rows)


if __name__ == "__main__":
    main()
