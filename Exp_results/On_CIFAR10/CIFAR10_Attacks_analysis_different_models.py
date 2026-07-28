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


SECTION_RE = re.compile(r"^\s*(?P<section>.+?summary across seeds):\s*$")
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
METHOD_STYLES = {
    "RFU": ("o", "-"),
    "Hessian": ("s", "--"),
    "Retraining": ("^", "-."),
}

PALETTE = {
    "RFU": "#797BB7",
    "Hessian": "#E07A5F",
    "Retraining": "#59A14F",
    "before": "#777777",
    "feature_overlap@1": "#797BB7",
    "feature_overlap@5": "#86BBD8",
    "feature_overlap@10": "#E1C855",
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
        "axes.labelsize": 22,
        "axes.titlesize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 16,
        "legend.fontsize": 13,
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


def methods_in_rows(rows):
    present = {row["method"] for row in rows}
    return [method for method in METHOD_ORDER if method in present]


def rows_for_method(rows, method):
    return [row for row in rows if row["method"] == method]


def models_in_rows(rows):
    return sorted({row["model"] for row in rows}, key=model_sort_key)


def set_model_axis(ax, rows):
    models = models_in_rows(rows)
    ax.set_xlabel("Model Architecture")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models)
    if len(models) > 1:
        ax.set_xlim(-0.25, len(models) - 0.75)


def metric_values(method_rows, metric, models, scale=1.0):
    model_positions = {model: index for index, model in enumerate(models)}
    available_rows = [row for row in method_rows if metric in row]
    available_rows.sort(key=lambda row: model_sort_key(row["model"]))
    x = [model_positions[row["model"]] for row in available_rows]
    y = [row[metric] * scale for row in available_rows]
    std = [row.get(f"{metric}_std", 0.0) * scale for row in available_rows]
    return x, y, std


def baseline_rows(rows):
    """Choose one before-unlearning baseline per model, preferring RFU."""
    selected = []
    for model in models_in_rows(rows):
        candidates = [row for row in rows if row["model"] == model]
        candidates.sort(key=lambda row: (row["method"] != "RFU", METHOD_ORDER.index(row["method"])))
        if candidates:
            selected.append(candidates[0])
    return selected


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
    models = models_in_rows(rows)
    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    for method in methods_in_rows(rows):
        marker, linestyle = METHOD_STYLES[method]
        method_rows = rows_for_method(rows, method)
        x, y, std = metric_values(method_rows, "acc_after", models, scale=100.0)
        plot_with_band(ax, x, y, std, METHOD_LABELS[method], PALETTE[method], marker, linestyle)

    before_rows = baseline_rows(rows)
    if before_rows:
        x, y, _ = metric_values(before_rows, "acc_before", models, scale=100.0)
        ax.plot(x, y, color=PALETTE["before"], linestyle=":", linewidth=2.5, label="Before unlearning")

    set_model_axis(ax, rows)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_ylim(0, 95)
    style_legend(ax.legend(loc="best", fancybox=True))
    polish_axes(ax)
    save_figure(fig, f"{output_prefix}_utility")


def plot_mia(rows, output_prefix=OUTPUT_PREFIX):
    models = models_in_rows(rows)
    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    for method in methods_in_rows(rows):
        marker, linestyle = METHOD_STYLES[method]
        method_rows = rows_for_method(rows, method)
        x, y, std = metric_values(method_rows, "mia_acc_after", models, scale=100.0)
        plot_with_band(ax, x, y, std, f"{METHOD_LABELS[method]} after", PALETTE[method], marker, linestyle)

    before_rows = baseline_rows(rows)
    if before_rows:
        x, y, std = metric_values(before_rows, "mia_acc_before", models, scale=100.0)
        plot_with_band(ax, x, y, std, "Before unlearning", PALETTE["before"], "v", ":")

    ax.axhline(50, color="#555555", linestyle=":", linewidth=2.0)
    set_model_axis(ax, rows)
    ax.set_ylabel("MIA Accuracy (%)")
    ax.set_ylim(48, 62)
    style_legend(ax.legend(loc="best", fancybox=True))
    polish_axes(ax)
    save_figure(fig, f"{output_prefix}_mia_accuracy")


def plot_label_inference(rows, output_prefix=OUTPUT_PREFIX):
    models = models_in_rows(rows)
    metrics = [
        ("delta_z_overlap", "CrossLeak"),
        ("confidence_drop_overlap", "Confidence drop"),
        ("ulia_known_k_overlap", "ULIA known-k"),
        ("ulia_dynamic_overlap", "ULIA dynamic"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.0 * len(metrics), 4.8), sharey=True)
    for ax, (metric, title) in zip(axes, metrics):
        for method in methods_in_rows(rows):
            marker, linestyle = METHOD_STYLES[method]
            method_rows = rows_for_method(rows, method)
            x, y, std = metric_values(method_rows, metric, models, scale=100.0)
            plot_with_band(ax, x, y, std, METHOD_LABELS[method], PALETTE[method], marker, linestyle)
        ax.set_title(title)
        set_model_axis(ax, rows)
        ax.set_ylim(0, 105)
        legend = ax.legend(loc="best", fancybox=True, fontsize=10)
        style_legend(legend, alpha=0.72)
        polish_axes(ax)
    axes[0].set_ylabel("Label Infer. Overlap (%)")
    save_figure(fig, f"{output_prefix}_label_inference_overlap")


def plot_precision_recall(rows, output_prefix=OUTPUT_PREFIX):
    models = models_in_rows(rows)
    metrics = [
        ("delta_z", "CrossLeak"),
        ("confidence_drop", "Confidence drop"),
        ("ulia_known_k", "ULIA known-k"),
        ("ulia_dynamic", "ULIA dynamic"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.0 * len(metrics), 4.8), sharey=True)
    for ax, (prefix, title) in zip(axes, metrics):
        for method in methods_in_rows(rows):
            marker, linestyle = METHOD_STYLES[method]
            method_rows = rows_for_method(rows, method)
            x, precision, precision_std = metric_values(method_rows, f"{prefix}_precision", models, scale=100.0)
            _, recall, recall_std = metric_values(method_rows, f"{prefix}_recall", models, scale=100.0)
            plot_with_band(ax, x, precision, precision_std, f"{METHOD_LABELS[method]} Precision", PALETTE[method], marker, linestyle)
            plot_with_band(ax, x, recall, recall_std, f"{METHOD_LABELS[method]} Recall", PALETTE[method], marker, ":")
        ax.set_title(title)
        set_model_axis(ax, rows)
        ax.set_ylim(0, 105)
        polish_axes(ax)
    axes[0].set_ylabel("Precision / Recall (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig_legend = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.07), ncol=3, fancybox=True)
    style_legend(fig_legend)
    save_figure(fig, f"{output_prefix}_precision_recall", rect=(0, 0, 1, 0.90))


def plot_feature_overlap(rows, output_prefix=OUTPUT_PREFIX):
    models = models_in_rows(rows)
    methods = methods_in_rows(rows)
    fig, axes = plt.subplots(1, len(methods), figsize=(5.0 * len(methods), 4.8), sharey=True)
    if len(methods) == 1:
        axes = [axes]
    for ax, method in zip(axes, methods):
        method_rows = rows_for_method(rows, method)
        for metric, marker, linestyle in [
            ("feature_overlap@1", "o", "-"),
            ("feature_overlap@5", "s", "--"),
            ("feature_overlap@10", "^", "-."),
        ]:
            x, y, std = metric_values(method_rows, metric, models, scale=100.0)
            plot_with_band(ax, x, y, std, metric.replace("feature_", ""), PALETTE[metric], marker, linestyle)
        ax.set_title(METHOD_LABELS[method])
        set_model_axis(ax, rows)
        ax.set_ylim(0, 105)
        polish_axes(ax)
    axes[0].set_ylabel("Sensitive Feature Overlap (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig_legend = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True)
    style_legend(fig_legend)
    save_figure(fig, f"{output_prefix}_feature_overlap", rect=(0, 0, 1, 0.92))


def plot_feature_overlap_bar(rows, output_prefix=OUTPUT_PREFIX):
    metrics = [
        ("feature_overlap@1", "overlap@1"),
        ("feature_overlap@5", "overlap@5"),
        ("feature_overlap@10", "overlap@10"),
    ]
    configurations = sorted(
        rows,
        key=lambda row: (model_sort_key(row["model"]), METHOD_ORDER.index(row["method"])),
    )
    x_positions = list(range(len(configurations)))
    bar_width = 0.22

    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    for metric_index, (metric, label) in enumerate(metrics):
        offsets = [
            position + (metric_index - (len(metrics) - 1) / 2) * bar_width
            for position in x_positions
        ]
        values = [row.get(metric, 0.0) * 100.0 for row in configurations]
        errors = [row.get(f"{metric}_std", 0.0) * 100.0 for row in configurations]
        ax.bar(
            offsets,
            values,
            yerr=errors,
            width=bar_width,
            label=label,
            color=PALETTE[metric],
            edgecolor="#222222",
            linewidth=0.8,
            alpha=0.92,
            capsize=3,
        )

    ax.set_xlabel("Model Architecture / Unlearning Method")
    ax.set_ylabel("Sensitive Feature Overlap (%)")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([
        f"{row['model']}\n{METHOD_LABELS[row['method']]}" for row in configurations
    ])
    ax.set_ylim(0, 105)
    style_legend(ax.legend(loc="best", fancybox=True))
    polish_axes(ax)
    save_figure(fig, f"{output_prefix}_feature_overlap_bar")


def plot_feature_and_clip(rows, output_prefix=OUTPUT_PREFIX):
    models = models_in_rows(rows)
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.8), sharey=True)

    for method in methods_in_rows(rows):
        marker, linestyle = METHOD_STYLES[method]
        method_rows = rows_for_method(rows, method)
        x, y, std = metric_values(method_rows, "before_exclusive_strength_ratio", models, scale=100.0)
        plot_with_band(axes[0], x, y, std, METHOD_LABELS[method], PALETTE[method], marker, linestyle)
        x, y, std = metric_values(method_rows, "target_label_feature_coverage", models, scale=100.0)
        plot_with_band(axes[1], x, y, std, METHOD_LABELS[method], PALETTE[method], marker, linestyle)

    axes[0].set_title("Before-exclusive Strength")
    set_model_axis(axes[0], rows)
    axes[0].set_ylabel("Metric Value (%)")
    axes[0].set_ylim(0, 105)
    polish_axes(axes[0])

    axes[1].set_title("Target Feature Coverage")
    set_model_axis(axes[1], rows)
    axes[1].set_ylim(0, 105)
    polish_axes(axes[1])

    for method in methods_in_rows(rows):
        marker, linestyle = METHOD_STYLES[method]
        method_rows = rows_for_method(rows, method)
        x, y, std = metric_values(method_rows, "clip_concept_coverage", models, scale=100.0)
        plot_with_band(axes[2], x, y, std, METHOD_LABELS[method], PALETTE[method], marker, linestyle, linewidth=2.6)

    axes[2].set_title("CLIP Concept Coverage")
    set_model_axis(axes[2], rows)
    axes[2].set_ylim(0, 105)
    polish_axes(axes[2])

    for ax in axes:
        legend = ax.legend(loc="best", fancybox=True, fontsize=10)
        style_legend(legend, alpha=0.72)

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
    plot_feature_overlap_bar(rows)
    plot_feature_and_clip(rows)
    print_summary(rows)


if __name__ == "__main__":
    main()
