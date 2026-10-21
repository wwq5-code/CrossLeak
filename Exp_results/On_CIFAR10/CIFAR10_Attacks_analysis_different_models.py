import csv
import math
import os
import re
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
RESULTS_FILE = BASE_DIR / "results_different_models.txt"
FIG_DIR = BASE_DIR / "figures_attacks_different_models"
TABLE_DIR = BASE_DIR / "tables_attacks_different_models"
OUTPUT_PREFIX = "CIFAR10_Attacks_different_models"
TABLE_NAME = "parsed_cifar10_attack_different_models_results.csv"

os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".mplconfig"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt


SECTION_RE = re.compile(r"^\s*(?P<section>.+?)\s+summary across seeds:\s*$", re.IGNORECASE)
METRIC_RE = re.compile(
    r"^\s*(?P<metric>[A-Za-z0-9_@]+):\s*"
    r"mean=(?P<mean>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?),\s*"
    r"variance=(?P<variance>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?)\s*$"
)


METHOD_LABELS = {
    "RFU": "FedU",
    "Hessian": "Hessian-FU",
    "Retraining": "FL Retraining",
}
METHOD_ORDER = ["RFU", "Hessian", "Retraining"]
PALETTE = {
    "RFU": "#797BB7",
    "Hessian": "#E07A5F",
    "Retraining": "#59A14F",
    "before": "#A9ADB4",
    "after": "#4C78A8",
    "precision": "#4C78A8",
    "recall": "#F28E2B",
    "feature_overlap@1": "#797BB7",
    "feature_overlap@5": "#86BBD8",
    "feature_overlap@10": "#E1C855",
    "before_exclusive_strength_ratio": "#797BB7",
    "target_label_feature_coverage": "#59A14F",
    "clip_concept_coverage": "#E07A5F",
}

EXCLUDED_METRIC_PREFIXES = ("hybrid", "clip_weighted_coverage")


def model_sort_key(model):
    preferred_order = {"ResNet18": 0, "VGG": 1}
    return preferred_order.get(model, len(preferred_order)), model.lower()


def method_and_model_from_section(section):
    section = section.strip()
    patterns = [
        ("Retraining", r"^FL retraining\s+(?P<model>.+)$"),
        ("Hessian", r"^Hessian\s+(?P<model>.+?)\s+unlearning$"),
        ("RFU", r"^RFU\s+(?P<model>.+)$"),
    ]
    for method, pattern in patterns:
        match = re.match(pattern, section, flags=re.IGNORECASE)
        if match:
            return method, match.group("model").strip()
    return None, None


def parse_results(results_file=RESULTS_FILE):
    rows = []
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
                method, model = method_and_model_from_section(section_match.group("section"))
                if method is not None:
                    current_row = {
                        "method": method,
                        "model": model,
                    }
                else:
                    current_row = None
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
        raise RuntimeError(f"No different-model summary rows parsed from {results_file}")
    return sorted(rows, key=lambda row: (model_sort_key(row["model"]), METHOD_ORDER.index(row["method"])))


def configure_plot_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.linewidth": 1.0,
        "axes.edgecolor": "#333333",
        "grid.color": "#D7D7D7",
        "grid.alpha": 0.75,
        "grid.linestyle": "--",
        "grid.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def polish_axes(ax):
    ax.grid(True, which="major", axis="y")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")


def style_legend(legend, alpha=0.92):
    if legend is None:
        return
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_alpha(alpha)
    frame.set_edgecolor("#BBBBBB")
    frame.set_linewidth(0.7)


def methods_in_rows(rows):
    present = {row["method"] for row in rows}
    return [method for method in METHOD_ORDER if method in present]


def rows_for_method(rows, method):
    return [row for row in rows if row["method"] == method]


def models_in_rows(rows):
    return sorted({row["model"] for row in rows}, key=model_sort_key)


def configuration_rows(rows):
    return sorted(
        rows,
        key=lambda row: (model_sort_key(row["model"]), METHOD_ORDER.index(row["method"])),
    )


def configuration_labels(rows):
    return [
        f"{row['model']}\n{METHOD_LABELS[row['method']]}"
        for row in configuration_rows(rows)
    ]


def set_configuration_axis(ax, rows, show_labels=True):
    positions = list(range(len(configuration_rows(rows))))
    ax.set_xticks(positions)
    ax.set_xticklabels(configuration_labels(rows) if show_labels else [])
    ax.set_xlim(-0.55, len(positions) - 0.45)


def draw_grouped_bars(ax, rows, series, ylim=(0, 105), show_labels=True):
    """Draw metric series for each (model, method) configuration."""
    ordered_rows = configuration_rows(rows)
    positions = list(range(len(ordered_rows)))
    width = min(0.72 / max(len(series), 1), 0.30)
    containers = []

    for index, (metric, label, color) in enumerate(series):
        offset = (index - (len(series) - 1) / 2.0) * width
        values = [row.get(metric, 0.0) * 100.0 for row in ordered_rows]
        errors = [row.get(f"{metric}_std", 0.0) * 100.0 for row in ordered_rows]
        container = ax.bar(
            [position + offset for position in positions],
            values,
            yerr=errors,
            width=width * 0.92,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            capsize=2.5,
            error_kw={"elinewidth": 1.0, "capthick": 1.0, "ecolor": "#444444"},
            zorder=3,
        )
        containers.append(container)

    set_configuration_axis(ax, rows, show_labels=show_labels)
    ax.set_ylim(*ylim)
    polish_axes(ax)
    return containers


def draw_single_metric_bars(ax, rows, metric, ylim=(0, 105), show_labels=True):
    ordered_rows = configuration_rows(rows)
    values = [row.get(metric, 0.0) * 100.0 for row in ordered_rows]
    errors = [row.get(f"{metric}_std", 0.0) * 100.0 for row in ordered_rows]
    colors = [PALETTE[row["method"]] for row in ordered_rows]
    ax.bar(
        range(len(ordered_rows)),
        values,
        yerr=errors,
        width=0.62,
        color=colors,
        edgecolor="white",
        linewidth=0.7,
        capsize=2.5,
        error_kw={"elinewidth": 1.0, "capthick": 1.0, "ecolor": "#444444"},
        zorder=3,
    )
    set_configuration_axis(ax, rows, show_labels=show_labels)
    ax.set_ylim(*ylim)
    polish_axes(ax)


def save_figure(fig, name, rect=None):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=rect)
    fig.savefig(FIG_DIR / f"{name}.pdf", format="pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_tables(rows):
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    metric_names = sorted({
        key
        for row in rows
        for key in row
        if key not in {"method", "model"} and not key.endswith("_std") and not key.endswith("_variance")
        and not key.startswith(EXCLUDED_METRIC_PREFIXES)
    })
    fieldnames = ["method", "model"]
    for metric in metric_names:
        fieldnames.extend([metric, f"{metric}_std", f"{metric}_variance"])

    with (TABLE_DIR / TABLE_NAME).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def plot_utility(rows, output_prefix=OUTPUT_PREFIX):
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    draw_grouped_bars(
        ax,
        rows,
        [
            ("acc_before", "Before unlearning", PALETTE["before"]),
            ("acc_after", "After unlearning", PALETTE["after"]),
        ],
        ylim=(0, 100),
    )
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Model Utility Before and After Unlearning", pad=10)
    style_legend(ax.legend(loc="upper right", ncol=2))
    save_figure(fig, f"{output_prefix}_utility")


def plot_mia(rows, output_prefix=OUTPUT_PREFIX):
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    draw_grouped_bars(
        ax,
        rows,
        [
            ("mia_acc_before", "Before unlearning", PALETTE["before"]),
            ("mia_acc_after", "After unlearning", PALETTE["after"]),
        ],
        ylim=(48, 62),
    )
    ax.axhline(50, color="#666666", linestyle=":", linewidth=1.2, label="Random guess")
    ax.set_ylabel("MIA Accuracy (%)")
    ax.set_title("Membership Inference Accuracy", pad=10)
    style_legend(ax.legend(loc="upper right", ncol=3))
    save_figure(fig, f"{output_prefix}_mia_accuracy")


def plot_label_inference(rows, output_prefix=OUTPUT_PREFIX):
    metrics = [
        ("delta_z_overlap", "CrossLeak"),
        ("confidence_drop_overlap", "Confidence Drop"),
        ("ulia_known_k_overlap", "ULIA (known k)"),
        ("ulia_dynamic_overlap", "ULIA (dynamic)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), sharex=True, sharey=True)
    for index, (ax, (metric, title)) in enumerate(zip(axes.flat, metrics)):
        draw_single_metric_bars(ax, rows, metric, show_labels=index >= 2)
        ax.set_title(title, pad=8)
        if index % 2 == 0:
            ax.set_ylabel("Label Inference Overlap (%)")
    save_figure(fig, f"{output_prefix}_label_inference_overlap")


def plot_precision_recall(rows, output_prefix=OUTPUT_PREFIX):
    metrics = [
        ("delta_z", "CrossLeak"),
        ("confidence_drop", "Confidence Drop"),
        ("ulia_known_k", "ULIA (known k)"),
        ("ulia_dynamic", "ULIA (dynamic)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), sharex=True, sharey=True)
    for index, (ax, (prefix, title)) in enumerate(zip(axes.flat, metrics)):
        draw_grouped_bars(
            ax,
            rows,
            [
                (f"{prefix}_precision", "Precision", PALETTE["precision"]),
                (f"{prefix}_recall", "Recall", PALETTE["recall"]),
            ],
            show_labels=index >= 2,
        )
        ax.set_title(title, pad=8)
        if index % 2 == 0:
            ax.set_ylabel("Score (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig_legend = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=2)
    style_legend(fig_legend)
    save_figure(fig, f"{output_prefix}_precision_recall", rect=(0, 0, 1, 0.95))


def plot_feature_overlap(rows, output_prefix=OUTPUT_PREFIX):
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    draw_grouped_bars(
        ax,
        rows,
        [
            ("feature_overlap@1", "Overlap@1", PALETTE["feature_overlap@1"]),
            ("feature_overlap@5", "Overlap@5", PALETTE["feature_overlap@5"]),
            ("feature_overlap@10", "Overlap@10", PALETTE["feature_overlap@10"]),
        ],
    )
    ax.set_ylabel("Sensitive Feature Overlap (%)")
    ax.set_title("Top-k Sensitive Feature Recovery", pad=10)
    style_legend(ax.legend(loc="upper left", ncol=3))
    save_figure(fig, f"{output_prefix}_feature_overlap")


def plot_rfu_model_comparison(rows, output_prefix=OUTPUT_PREFIX):
    rfu_rows = [row for row in rows if row["method"] == "RFU"]
    if not rfu_rows:
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    draw_grouped_bars(
        ax,
        rfu_rows,
        [
            ("feature_overlap@1", "Overlap@1", PALETTE["feature_overlap@1"]),
            ("feature_overlap@5", "Overlap@5", PALETTE["feature_overlap@5"]),
            ("feature_overlap@10", "Overlap@10", PALETTE["feature_overlap@10"]),
        ],
    )
    ax.set_xticklabels([row["model"] for row in configuration_rows(rfu_rows)])
    ax.set_xlabel("Model Architecture")
    ax.set_ylabel("Sensitive Feature Overlap (%)")
    ax.set_title("FedU Cross-Architecture Comparison", pad=10)
    style_legend(ax.legend(loc="upper center", ncol=3))
    save_figure(fig, f"{output_prefix}_rfu_model_comparison")


def plot_feature_and_clip(rows, output_prefix=OUTPUT_PREFIX):
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    draw_grouped_bars(
        ax,
        rows,
        [
            ("before_exclusive_strength_ratio", "Exclusive strength", PALETTE["before_exclusive_strength_ratio"]),
            ("target_label_feature_coverage", "Target feature coverage", PALETTE["target_label_feature_coverage"]),
            ("clip_concept_coverage", "CLIP concept coverage", PALETTE["clip_concept_coverage"]),
        ],
    )
    ax.set_ylabel("Metric Value (%)")
    ax.set_title("Sensitive Feature and Semantic Coverage", pad=10)
    style_legend(ax.legend(loc="upper center", ncol=3))
    save_figure(fig, f"{output_prefix}_feature_and_clip")


def print_summary(rows):
    print("CIFAR-10 different-model attack analysis parsed from results_different_models.txt")
    for method in methods_in_rows(rows):
        method_rows = rows_for_method(rows, method)
        avg_acc_after = sum(row["acc_after"] for row in method_rows) / len(method_rows)
        avg_feature5 = sum(row["feature_overlap@5"] for row in method_rows) / len(method_rows)
        avg_clip = sum(row["clip_concept_coverage"] for row in method_rows) / len(method_rows)
        print(
            f"{METHOD_LABELS[method]}: "
            f"models={[row['model'] for row in method_rows]}, "
            f"avg acc_after={avg_acc_after:.4f}, "
            f"avg feature_overlap@5={avg_feature5:.4f}, "
            f"avg clip_concept_coverage={avg_clip:.4f}"
        )
    print(f"Figures will be saved to: {FIG_DIR}")
    print(f"Tables will be saved to: {TABLE_DIR}")


def main():
    configure_plot_style()
    rows = parse_results(RESULTS_FILE)
    save_tables(rows)
    plot_utility(rows)
    plot_mia(rows)
    plot_label_inference(rows)
    plot_precision_recall(rows)
    plot_feature_overlap(rows)
    plot_rfu_model_comparison(rows)
    plot_feature_and_clip(rows)
    print_summary(rows)


if __name__ == "__main__":
    main()
