#!/usr/bin/env python3
"""Visualize saturation in the complete Amazon 112-label results."""

from __future__ import annotations

import argparse
import csv
import html
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import PercentFormatter


MODEL_ORDER = ("Qwen3.5-9B", "Llama-3.1-8B-Instruct")
TRAIN_SIZES_K = (5, 10, 20, 40, 60, 80, 100, 112)
METRICS = (
    ("accuracy", "Macro Accuracy", "Accuracy (%)"),
    ("recall_10", "Macro Recall@10", "Recall@10"),
    ("recall_20", "Macro Recall@20", "Recall@20"),
    ("ndcg_10", "Macro nDCG@10", "nDCG@10"),
    ("ndcg_20", "Macro nDCG@20", "nDCG@20"),
)
RANKING_METRICS = tuple(metric[0] for metric in METRICS[1:])
MODEL_STYLES = {
    "Qwen3.5-9B": {"color": "#0072B2", "marker": "o", "linestyle": "-"},
    "Llama-3.1-8B-Instruct": {
        "color": "#D55E00",
        "marker": "s",
        "linestyle": "--",
    },
}
PLATEAU_RATIO = 0.95
RANKING_PLATEAU_K = 80


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(__file__).with_name("RESULTS.md"),
        help="112-label RESULTS.md to parse.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("figures"),
        help="Directory for the generated figure and diagnostic CSV.",
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def parse_train_size(value: str) -> int:
    match = re.fullmatch(r"(\d+)K(?: \(all\))?", value)
    if not match:
        raise ValueError(f"invalid Train rows value: {value!r}")
    return int(match.group(1))


def parse_results(path: Path) -> dict[str, dict[str, list[float]]]:
    cell_pattern = re.compile(r"<td(?:\s+rowspan=\"\d+\")?>(.*?)</td>")
    parsed: dict[str, dict[str, list[float]]] = {
        model: {"train_size_k": []} | {key: [] for key, _, _ in METRICS}
        for model in MODEL_ORDER
    }
    current_model: str | None = None

    for line in path.read_text(encoding="utf-8").splitlines():
        cells = [html.unescape(value.strip()) for value in cell_pattern.findall(line)]
        if not cells:
            continue
        if cells[0] in MODEL_ORDER:
            current_model = cells.pop(0)
        if current_model is None or len(cells) != 6:
            raise ValueError(f"unexpected result row in {path}: {line.strip()}")

        train_size, accuracy, *ranking_values = cells
        model_data = parsed[current_model]
        model_data["train_size_k"].append(parse_train_size(train_size))
        model_data["accuracy"].append(float(accuracy.removesuffix("%")))
        for metric, value in zip(RANKING_METRICS, ranking_values, strict=True):
            model_data[metric].append(float(value))

    for model, model_data in parsed.items():
        sizes = tuple(int(value) for value in model_data["train_size_k"])
        if sizes != TRAIN_SIZES_K:
            raise ValueError(
                f"{model} has Train sizes {sizes}; expected complete {TRAIN_SIZES_K}"
            )
    return parsed


def stable_onset_k(
    train_sizes: list[float], values: list[float], ratio: float = PLATEAU_RATIO
) -> int | None:
    """Return the earliest sampled size whose suffix stays near the endpoint."""
    threshold = values[-1] * ratio
    for index, train_size in enumerate(train_sizes[:-1]):
        if all(value >= threshold for value in values[index:]):
            return int(train_size)
    return None


def saturation_rows(
    results: dict[str, dict[str, list[float]]],
) -> list[dict[str, str | float | int]]:
    rows: list[dict[str, str | float | int]] = []
    for model in MODEL_ORDER:
        model_data = results[model]
        train_sizes = model_data["train_size_k"]
        for metric, metric_title, _ in METRICS:
            values = model_data[metric]
            onset = stable_onset_k(train_sizes, values)
            value_80k = values[train_sizes.index(RANKING_PLATEAU_K)]
            endpoint = values[-1]
            rows.append(
                {
                    "model": model,
                    "metric": metric_title,
                    "stable_95pct_onset_k": onset if onset is not None else "not_observed",
                    "value_at_80k": value_80k,
                    "value_at_112k": endpoint,
                    "ratio_80k_to_112k_pct": 100 * value_80k / endpoint,
                    "change_80k_to_112k": endpoint - value_80k,
                }
            )
    return rows


def write_diagnostics(
    path: Path, rows: list[dict[str, str | float | int]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def style_metric_axis(axis: Axes, metric: str) -> None:
    axis.set_xticks(TRAIN_SIZES_K)
    axis.set_xlim(2, 115)
    axis.grid(axis="y", color="#D7DCE1", linewidth=0.8, alpha=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(colors="#3B4147")
    if metric == "accuracy":
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    else:
        axis.axvspan(
            RANKING_PLATEAU_K,
            TRAIN_SIZES_K[-1],
            color="#DDEFE3",
            alpha=0.8,
            zorder=0,
        )
        axis.axvline(
            RANKING_PLATEAU_K,
            color="#2E7D4F",
            linestyle=":",
            linewidth=1.5,
            zorder=1,
        )


def plot_metric(
    axis: Axes,
    results: dict[str, dict[str, list[float]]],
    metric: str,
    title: str,
    ylabel: str,
) -> None:
    for model in MODEL_ORDER:
        model_data = results[model]
        axis.plot(
            model_data["train_size_k"],
            model_data[metric],
            label=model,
            linewidth=2.2,
            markersize=6.5,
            markeredgecolor="white",
            markeredgewidth=0.8,
            **MODEL_STYLES[model],
        )
    style_metric_axis(axis, metric)
    axis.set_title(title, loc="left", fontsize=11.5, fontweight="bold")
    axis.set_xlabel("Training rows (K)")
    axis.set_ylabel(ylabel)

    if metric == "accuracy":
        axis.axvspan(80, 112, color="#FCE8DD", alpha=0.72, zorder=0)
        axis.text(
            0.98,
            0.06,
            "No saturation observed\n+5.55 / +4.26 pp after 80K",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            color="#8C3B14",
            fontsize=9,
            fontweight="bold",
        )
    elif metric == "recall_10":
        axis.text(
            82,
            axis.get_ylim()[0] + 0.08 * np.ptp(axis.get_ylim()),
            "ranking plateau\nregion (~80K+)",
            color="#2E6B45",
            fontsize=8.5,
            fontweight="bold",
        )


def plot_endpoint_diagnostic(
    axis: Axes, rows: list[dict[str, str | float | int]]
) -> None:
    metric_labels = ("Accuracy", "R@10", "R@20", "nDCG@10", "nDCG@20")
    positions = np.arange(len(metric_labels))
    width = 0.36
    axis.axhspan(95, 106, color="#DDEFE3", alpha=0.8, zorder=0)
    axis.axhline(95, color="#2E7D4F", linestyle=":", linewidth=1.5)

    for offset, model in zip((-width / 2, width / 2), MODEL_ORDER, strict=True):
        model_rows = [row for row in rows if row["model"] == model]
        values = [float(row["ratio_80k_to_112k_pct"]) for row in model_rows]
        bars = axis.bar(
            positions + offset,
            values,
            width,
            label=model,
            color=MODEL_STYLES[model]["color"],
            alpha=0.9,
            edgecolor="white",
            linewidth=0.7,
        )
        axis.bar_label(bars, fmt="%.1f", padding=2, fontsize=7.5)

    axis.set_ylim(65, 106)
    axis.set_xticks(positions, metric_labels, rotation=18, ha="right")
    axis.set_ylabel("80K / 112K endpoint (%)")
    axis.set_title("Saturation diagnostic", loc="left", fontsize=11.5, fontweight="bold")
    axis.grid(axis="y", color="#D7DCE1", linewidth=0.8, alpha=0.8)
    axis.spines[["top", "right"]].set_visible(False)


def plot_saturation(
    path: Path,
    results: dict[str, dict[str, list[float]]],
    rows: list[dict[str, str | float | int]],
    dpi: int,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.labelcolor": "#30363B",
            "text.color": "#20262B",
        }
    )
    figure, axes = plt.subplots(2, 3, figsize=(15.5, 9.2))
    figure.patch.set_facecolor("#FAFBFC")
    for axis in axes.flat:
        axis.set_facecolor("#FFFFFF")

    for axis, (metric, title, ylabel) in zip(axes.flat[:5], METRICS, strict=True):
        plot_metric(axis, results, metric, title, ylabel)
    plot_endpoint_diagnostic(axes.flat[5], rows)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=2,
        frameon=False,
        fontsize=10.5,
    )
    figure.suptitle(
        "Amazon 112-label training-data saturation",
        x=0.055,
        y=0.985,
        ha="left",
        fontsize=18,
        fontweight="bold",
    )
    figure.text(
        0.055,
        0.955,
        "Raw single-run points; no smoothing. Ranking reaches an empirical plateau around 80K, while accuracy is still improving.",
        ha="left",
        fontsize=10.5,
        color="#4B545B",
    )
    figure.text(
        0.055,
        0.015,
        "Plateau criterion: earliest sampled size from which every later point remains >=95% of the 112K endpoint. "
        "A final-point-only match is reported as not observed; this is an empirical plateau, not a statistical breakpoint.",
        ha="left",
        fontsize=8.8,
        color="#525C63",
    )
    figure.subplots_adjust(left=0.06, right=0.985, top=0.89, bottom=0.09, hspace=0.37, wspace=0.28)
    figure.savefig(path, dpi=dpi, facecolor=figure.get_facecolor())
    plt.close(figure)


def main() -> None:
    args = parse_args()
    results = parse_results(args.results)
    rows = saturation_rows(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_diagnostics(args.output_dir / "saturation_diagnostics.csv", rows)
    plot_saturation(args.output_dir / "saturation_curves.png", results, rows, args.dpi)
    plot_saturation(args.output_dir / "saturation_curves.pdf", results, rows, args.dpi)

    ranking_onsets = {
        model: max(
            int(row["stable_95pct_onset_k"])
            for row in rows
            if row["model"] == model and row["metric"] != "Macro Accuracy"
        )
        for model in MODEL_ORDER
    }
    print(f"Wrote saturation artifacts to {args.output_dir}")
    for model in MODEL_ORDER:
        print(
            f"{model}: all ranking metrics stable within 95% of endpoint from "
            f"{ranking_onsets[model]}K; accuracy saturation not observed"
        )


if __name__ == "__main__":
    main()