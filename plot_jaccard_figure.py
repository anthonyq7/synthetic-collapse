# plot_jaccard_figure.py
# Single-panel Jaccard overlap figure.
#
# Run (from repo root):
#   pip install matplotlib numpy
#   python plot_jaccard_figure.py --out artifacts/nodewise_jaccard.png
#
# Inputs (expected paths):
#   artifacts/gpt_theory_metrics_jaccard.csv
#   artifacts/claude_theory_metrics_jaccard.csv
#   artifacts/gpt_random_jaccard_agg.csv
#   artifacts/claude_random_jaccard_agg.csv

# plot_jaccard_figure.py
# Run:
#   python plot_jaccard_figure.py --out artifacts/nodewise_jaccard.png

from __future__ import annotations

import argparse
import os
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt

MAX_NODES_DEFAULT = 12


def load_csv_as_dict_by_node(path: str) -> Dict[int, dict]:
    import csv

    out = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            out[int(r["node"])] = r
    return out


def float_or_nan(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def ensure_paths_exist(paths):
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError("Missing required file(s):\n  " + "\n  ".join(missing))


def plot_baseline_with_ci(ax, x, mean, lo, hi, *, label, color, alpha=0.12):
    ax.plot(x, mean, color=color, linestyle="--", linewidth=2, label=label)
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, linewidth=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/nodewise_jaccard.png")
    ap.add_argument("--max-nodes", type=int, default=MAX_NODES_DEFAULT)
    args = ap.parse_args()

    GPT_J = "artifacts/gpt_theory_metrics_jaccard.csv"
    CLAUDE_J = "artifacts/claude_theory_metrics_jaccard.csv"
    GPT_J_R = "artifacts/gpt_random_jaccard_agg.csv"
    CLAUDE_J_R = "artifacts/claude_random_jaccard_agg.csv"

    ensure_paths_exist([GPT_J, CLAUDE_J, GPT_J_R, CLAUDE_J_R])

    gpt = load_csv_as_dict_by_node(GPT_J)
    claude = load_csv_as_dict_by_node(CLAUDE_J)
    gpt_r = load_csv_as_dict_by_node(GPT_J_R)
    claude_r = load_csv_as_dict_by_node(CLAUDE_J_R)

    nodes = list(range(args.max_nodes))
    x = np.arange(len(nodes))

    GPT_COLOR = "#04D8A3"
    CLAUDE_COLOR = "#D97757"

    gpt_y = [float_or_nan(gpt[n]["mean_jaccard_O_pq_t"]) for n in nodes]
    claude_y = [float_or_nan(claude[n]["mean_jaccard_O_pq_t"]) for n in nodes]

    gpt_m = [float_or_nan(gpt_r[n]["jaccard_mean"]) for n in nodes]
    gpt_lo = [float_or_nan(gpt_r[n]["jaccard_ci95_low"]) for n in nodes]
    gpt_hi = [float_or_nan(gpt_r[n]["jaccard_ci95_high"]) for n in nodes]

    claude_m = [float_or_nan(claude_r[n]["jaccard_mean"]) for n in nodes]
    claude_lo = [float_or_nan(claude_r[n]["jaccard_ci95_low"]) for n in nodes]
    claude_hi = [float_or_nan(claude_r[n]["jaccard_ci95_high"]) for n in nodes]

    fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)

    ax.plot(x, gpt_y, color=GPT_COLOR, marker="o", linewidth=2, label="GPT-5 mini")
    ax.plot(x, claude_y, color=CLAUDE_COLOR, marker="o", linewidth=2, label="Claude Haiku 4.5")

    plot_baseline_with_ci(
        ax, x, gpt_m, gpt_lo, gpt_hi,
        label="GPT matched-random baseline (mean ± CI)", color=GPT_COLOR, alpha=0.10
    )
    plot_baseline_with_ci(
        ax, x, claude_m, claude_lo, claude_hi,
        label="Claude matched-random baseline (mean ± CI)", color=CLAUDE_COLOR, alpha=0.10
    )

    ax.set_title("Mean prompt-pair Jaccard overlap by node", fontsize=12)
    ax.set_xlabel("Node", fontsize=10)
    ax.set_ylabel("Mean Jaccard(C, C')", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(nodes, fontsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=9, loc="upper left")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=220, bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()