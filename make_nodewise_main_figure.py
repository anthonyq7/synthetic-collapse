# make_nodewise_main_figure.py
# Run (from repo root):
#   pip install matplotlib numpy
#   python make_nodewise_main_figure.py --out figures/nodewise_main.png
#
# Inputs (expected paths):
#   Top-share:
#     gpt/output/master/concentration.jsonl
#     gpt/output/master/concentration_random_baseline.jsonl
#     claude/output/master/concentration.jsonl
#     claude/output/master/concentration_random_baseline.jsonl
#   Exclusion Rates:
#     gpt/output/analysis_figures/exclusion_rate_by_node.jsonl
#     gpt/output/master/false_negatives_random_baseline.jsonl
#     claude/output/analysis_figures/exclusion_rate_by_node.jsonl
#     claude/output/master/false_negatives_random_baseline.jsonl
#   Theory-facing HHI + raw overlap:
#     artifacts/gpt_theory_metrics.csv
#     artifacts/claude_theory_metrics.csv
#     artifacts/gpt_random_theory_agg.csv
#     artifacts/claude_random_theory_agg.csv

# make_nodewise_main_figure.py

# Run:
#   python make_nodewise_main_figure.py --out figures/nodewise_main.png

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt

import matplotlib.ticker as mticker

MAX_NODES_DEFAULT = 12


def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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


def ensure_paths_exist(paths: List[str]):
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError("Missing required file(s):\n  " + "\n  ".join(missing))


def plot_baseline_with_ci(ax, x, mean, lo, hi, *, label, color, alpha=0.32):
    # dashed baseline mean in same color; shaded CI in same color
    ax.plot(x, mean, color=color, linestyle="--", linewidth=2, label=label)
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, linewidth=0)


def style_ax(ax, title, ylabel):
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Node", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="both", labelsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/nodewise_main.png")
    ap.add_argument("--max-nodes", type=int, default=MAX_NODES_DEFAULT)
    args = ap.parse_args()

    GPT_CONC = "gpt/output/master/concentration.jsonl"
    GPT_CONC_RND = "gpt/output/master/concentration_random_baseline.jsonl"
    CLAUDE_CONC = "claude/output/master/concentration.jsonl"
    CLAUDE_CONC_RND = "claude/output/master/concentration_random_baseline.jsonl"

    GPT_FN = "gpt/output/analysis_figures/exclusion_rate_by_node.jsonl"
    GPT_FN_RND = "gpt/output/master/false_negatives_random_baseline.jsonl"
    CLAUDE_FN = "claude/output/analysis_figures/exclusion_rate_by_node.jsonl"
    CLAUDE_FN_RND = "claude/output/master/false_negatives_random_baseline.jsonl"

    GPT_THEORY = "artifacts/gpt_theory_metrics.csv"
    CLAUDE_THEORY = "artifacts/claude_theory_metrics.csv"
    GPT_THEORY_RND = "artifacts/gpt_random_theory_agg.csv"
    CLAUDE_THEORY_RND = "artifacts/claude_random_theory_agg.csv"

    ensure_paths_exist(
        [
            GPT_CONC,
            GPT_CONC_RND,
            CLAUDE_CONC,
            CLAUDE_CONC_RND,
            GPT_FN,
            GPT_FN_RND,
            CLAUDE_FN,
            CLAUDE_FN_RND,
            GPT_THEORY,
            CLAUDE_THEORY,
            GPT_THEORY_RND,
            CLAUDE_THEORY_RND,
        ]
    )

    nodes = list(range(args.max_nodes))
    x = np.arange(len(nodes))

    # Consistent palette
    GPT_COLOR = "#04D8A3"  # green
    CLAUDE_COLOR = "#D97757"  # orange

    # ---------- Panel (a): Top-10% share ----------
    gpt_conc = {r["node"]: r for r in load_jsonl(GPT_CONC)}
    gpt_conc_r = {r["node"]: r for r in load_jsonl(GPT_CONC_RND)}
    claude_conc = {r["node"]: r for r in load_jsonl(CLAUDE_CONC)}
    claude_conc_r = {r["node"]: r for r in load_jsonl(CLAUDE_CONC_RND)}

    gpt_top = [float_or_nan(gpt_conc[n]["top_10pct_share"]) for n in nodes]
    claude_top = [float_or_nan(claude_conc[n]["top_10pct_share"]) for n in nodes]

    gpt_top_r_mean = [float_or_nan(gpt_conc_r[n]["top_10pct_share_mean"]) for n in nodes]
    gpt_top_r_lo = [float_or_nan(gpt_conc_r[n]["top_10pct_share_ci_low"]) for n in nodes]
    gpt_top_r_hi = [float_or_nan(gpt_conc_r[n]["top_10pct_share_ci_high"]) for n in nodes]

    claude_top_r_mean = [float_or_nan(claude_conc_r[n]["top_10pct_share_mean"]) for n in nodes]
    claude_top_r_lo = [float_or_nan(claude_conc_r[n]["top_10pct_share_ci_low"]) for n in nodes]
    claude_top_r_hi = [float_or_nan(claude_conc_r[n]["top_10pct_share_ci_high"]) for n in nodes]

    # ---------- Panel (b): Exclusion Rates ----------
    gpt_fn = {r["node"]: r for r in load_jsonl(GPT_FN)}
    gpt_fn_r = {r["node"]: r for r in load_jsonl(GPT_FN_RND)}
    claude_fn = {r["node"]: r for r in load_jsonl(CLAUDE_FN)}
    claude_fn_r = {r["node"]: r for r in load_jsonl(CLAUDE_FN_RND)}

    gpt_fn_y = [float_or_nan(gpt_fn[n]["shown_ignored_rate_of_shown"]) for n in nodes]
    claude_fn_y = [float_or_nan(claude_fn[n]["shown_ignored_rate_of_shown"]) for n in nodes]

    gpt_fn_r_mean = [float_or_nan(gpt_fn_r[n]["shown_ignored_rate_of_shown_mean"]) for n in nodes]
    gpt_fn_r_lo = [float_or_nan(gpt_fn_r[n]["shown_ignored_rate_of_shown_ci_low"]) for n in nodes]
    gpt_fn_r_hi = [float_or_nan(gpt_fn_r[n]["shown_ignored_rate_of_shown_ci_high"]) for n in nodes]

    claude_fn_r_mean = [float_or_nan(claude_fn_r[n]["shown_ignored_rate_of_shown_mean"]) for n in nodes]
    claude_fn_r_lo = [float_or_nan(claude_fn_r[n]["shown_ignored_rate_of_shown_ci_low"]) for n in nodes]
    claude_fn_r_hi = [float_or_nan(claude_fn_r[n]["shown_ignored_rate_of_shown_ci_high"]) for n in nodes]

    # ---------- Panel (c)(d): theory CSVs ----------
    gpt_theory = load_csv_as_dict_by_node(GPT_THEORY)
    claude_theory = load_csv_as_dict_by_node(CLAUDE_THEORY)
    gpt_theory_r = load_csv_as_dict_by_node(GPT_THEORY_RND)
    claude_theory_r = load_csv_as_dict_by_node(CLAUDE_THEORY_RND)

    gpt_hhi = [float_or_nan(gpt_theory[n]["HHI_t"]) for n in nodes]
    claude_hhi = [float_or_nan(claude_theory[n]["HHI_t"]) for n in nodes]

    gpt_hhi_r_mean = [float_or_nan(gpt_theory_r[n]["HHI_mean"]) for n in nodes]
    gpt_hhi_r_lo = [float_or_nan(gpt_theory_r[n]["HHI_ci95_low"]) for n in nodes]
    gpt_hhi_r_hi = [float_or_nan(gpt_theory_r[n]["HHI_ci95_high"]) for n in nodes]

    claude_hhi_r_mean = [float_or_nan(claude_theory_r[n]["HHI_mean"]) for n in nodes]
    claude_hhi_r_lo = [float_or_nan(claude_theory_r[n]["HHI_ci95_low"]) for n in nodes]
    claude_hhi_r_hi = [float_or_nan(claude_theory_r[n]["HHI_ci95_high"]) for n in nodes]

    gpt_ov = [float_or_nan(gpt_theory[n]["mean_overlap_O_pq_t"]) for n in nodes]
    claude_ov = [float_or_nan(claude_theory[n]["mean_overlap_O_pq_t"]) for n in nodes]

    gpt_ov_r_mean = [float_or_nan(gpt_theory_r[n]["overlap_mean"]) for n in nodes]
    gpt_ov_r_lo = [float_or_nan(gpt_theory_r[n]["overlap_ci95_low"]) for n in nodes]
    gpt_ov_r_hi = [float_or_nan(gpt_theory_r[n]["overlap_ci95_high"]) for n in nodes]

    claude_ov_r_mean = [float_or_nan(claude_theory_r[n]["overlap_mean"]) for n in nodes]
    claude_ov_r_lo = [float_or_nan(claude_theory_r[n]["overlap_ci95_low"]) for n in nodes]
    claude_ov_r_hi = [float_or_nan(claude_theory_r[n]["overlap_ci95_high"]) for n in nodes]

    # ---------- figure ----------
    fig, axs = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    # (a) Top-share
    ax = axs[0, 0]
    ax.plot(x, gpt_top, color=GPT_COLOR, marker="o", linewidth=2, label="GPT-5 mini")
    ax.plot(x, claude_top, color=CLAUDE_COLOR, marker="o", linewidth=2, label="Claude Haiku 4.5")
    plot_baseline_with_ci(
        ax, x, gpt_top_r_mean, gpt_top_r_lo, gpt_top_r_hi,
        label="GPT matched-random baseline (mean ± CI)", color=GPT_COLOR, alpha=0.30
    )
    plot_baseline_with_ci(
        ax, x, claude_top_r_mean, claude_top_r_lo, claude_top_r_hi,
        label="Claude matched-random baseline (mean ± CI)", color=CLAUDE_COLOR, alpha=0.30
    )
    style_ax(ax, "(a) Top-10% citation share by node", "Top 10% share (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(nodes)
    ax.legend(fontsize=8, loc="upper left")

    # (b) Exclusion Rates
    ax = axs[0, 1]
    ax.plot(x, gpt_fn_y, color=GPT_COLOR, marker="o", linewidth=2, label="GPT-5 mini")
    ax.plot(x, claude_fn_y, color=CLAUDE_COLOR, marker="o", linewidth=2, label="Claude Haiku 4.5")
    plot_baseline_with_ci(
        ax, x, gpt_fn_r_mean, gpt_fn_r_lo, gpt_fn_r_hi,
        label="GPT matched-random baseline (mean ± CI)", color=GPT_COLOR, alpha=0.30
    )
    plot_baseline_with_ci(
        ax, x, claude_fn_r_mean, claude_fn_r_lo, claude_fn_r_hi,
        label="Claude matched-random baseline (mean ± CI)", color=CLAUDE_COLOR, alpha=0.30
    )
    style_ax(ax, "(b) Exclusion Rates by node", "Shown-but-uncited / shown (rate)")
    ax.set_xticks(x)
    ax.set_xticklabels(nodes)
    ax.legend(fontsize=8, loc="upper left")

    # (c) HHI
    ax = axs[1, 0]
    ax.set_yscale("log")
    ax.yaxis.set_minor_locator(mticker.LogLocator(subs='auto'))
    ax.plot(x, gpt_hhi, color=GPT_COLOR, marker="o", linewidth=2, label="GPT-5 mini (HHI)")
    ax.plot(x, claude_hhi, color=CLAUDE_COLOR, marker="o", linewidth=2, label="Claude Haiku 4.5 (HHI)")
    plot_baseline_with_ci(
        ax, x, gpt_hhi_r_mean, gpt_hhi_r_lo, gpt_hhi_r_hi,
        label="GPT matched-random baseline (mean ± CI)", color=GPT_COLOR, alpha=0.60
    )
    plot_baseline_with_ci(
        ax, x, claude_hhi_r_mean, claude_hhi_r_lo, claude_hhi_r_hi,
        label="Claude matched-random baseline (mean ± CI)", color=CLAUDE_COLOR, alpha=0.60
    )
    style_ax(ax, "(c) HHI by node (theorem-facing)", "HHI")
    ax.set_xticks(x)
    ax.set_xticklabels(nodes)
    ax.legend(fontsize=8, loc="upper right")

    # (d) Overlap (raw count)
    ax = axs[1, 1]
    ax.set_yscale("log")
    ax.yaxis.set_minor_locator(mticker.LogLocator(subs='auto'))
    ax.plot(x, gpt_ov, color=GPT_COLOR, marker="o", linewidth=2, label="GPT-5 mini (mean |C∩C'|)")
    ax.plot(x, claude_ov, color=CLAUDE_COLOR, marker="o", linewidth=2, label="Claude Haiku 4.5 (mean |C∩C'|)")
    plot_baseline_with_ci(
        ax, x, gpt_ov_r_mean, gpt_ov_r_lo, gpt_ov_r_hi,
        label="GPT matched-random baseline (mean ± CI)", color=GPT_COLOR, alpha=0.60
    )
    plot_baseline_with_ci(
        ax, x, claude_ov_r_mean, claude_ov_r_lo, claude_ov_r_hi,
        label="Claude matched-random baseline (mean ± CI)", color=CLAUDE_COLOR, alpha=0.60
    )
    style_ax(ax, "(d) Bibliography overlap by node (theorem-facing)", "Mean pair overlap |C∩C'|")
    ax.set_xticks(x)
    ax.set_xticklabels(nodes)
    ax.legend(fontsize=8, loc="upper right")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=220, bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()