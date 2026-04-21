import json
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, MultipleLocator


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent

CONDITIONS = ["L1", "L2", "L1P"]
CONDITION_COLORS = {
    "L1": "#10A37F",
    "L2": "#D97757",
    "L1P": "#8E44AD",
    "L3": "#2E86DE",
}
# Short (on, off) lengths in points → denser dashes than default "--".
L1P_DENSE_DASHES = (0, (2.2, 1.6))
CONDITION_LINESTYLE = {
    "L1": "-",
    "L2": "-",
    "L1P": L1P_DENSE_DASHES,
    "L3": "-",
}
CONDITION_LEGEND = {
    "L1": "GPT",
    "L2": "GPT and Claude",
    "L1P": "GPT Placebo",
}

TOTAL_NODES = 12
SEED_COUNT = 120
PAPERS_PER_NODE = 120
CITATION_CAP = 10
TOP_PERCENT = 0.10

PRESAMPLED_KV_PATH = EXPERIMENT_DIR / "presampled" / "kv_pairs.jsonl"
CONCENTRATION_OUTPUT_PATH = SCRIPT_DIR / "concentration_by_node.jsonl"
OVERLAP_OUTPUT_PATH = SCRIPT_DIR / "raw_overlap_by_node.jsonl"
COMBINED_FIGURE_PATH = SCRIPT_DIR / "concentration_raw_overlap.png"


def load_valid_ids() -> Set[str]:
    valid_ids: Set[str] = set()
    with PRESAMPLED_KV_PATH.open("r") as kv_file:
        for raw_line in kv_file:
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            valid_ids.add(record["id"])
    return valid_ids


def _sort_key_count_desc(item: Tuple[str, int]) -> Tuple[int, str]:
    paper_id, count = item
    return (-count, paper_id)


def node_stats_path(condition: str, node: int) -> Path:
    return (
        EXPERIMENT_DIR
        / condition
        / "output"
        / f"node_{node}"
        / f"node_{node}_stats.jsonl"
    )


def node_papers_path(condition: str, node: int) -> Path:
    return (
        EXPERIMENT_DIR
        / condition
        / "output"
        / f"node_{node}"
        / f"node_{node}.jsonl"
    )


def load_node_stats(
    condition: str, node: int, valid_ids: Set[str]
) -> Tuple[Dict[str, int], int]:
    stats: Dict[str, int] = {}
    stray_count = 0
    path = node_stats_path(condition, node)
    with path.open("r") as stats_file:
        for raw_line in stats_file:
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            for cited_id, count in record.items():
                if cited_id in valid_ids:
                    stats[cited_id] = count
                else:
                    stray_count += 1
    return stats, stray_count


def top_n_share_fraction(
    sorted_stats: List[Tuple[str, int]], n: int, total_citations: int
) -> float:
    if total_citations <= 0:
        return 0.0
    top = sorted_stats[:n]
    top_citations = 0
    for _, count in top:
        top_citations += count
    return top_citations / total_citations


def top_percent_share(
    sorted_stats: List[Tuple[str, int]],
    percent: float,
    total_citations: int,
    available: int,
) -> Tuple[float, int]:
    if total_citations <= 0 or available <= 0 or percent <= 0.0:
        return 0.0, 0
    k = int(available * percent)
    if k < 1:
        k = 1
    k_effective = min(k, len(sorted_stats))
    frac = top_n_share_fraction(sorted_stats, k_effective, total_citations)
    return frac, k_effective


def available_papers(node: int) -> int:
    return SEED_COUNT + PAPERS_PER_NODE * node


def load_node_citation_sets(condition: str, node: int) -> List[Set[str]]:
    sets: List[Set[str]] = []
    path = node_papers_path(condition, node)
    with path.open("r") as papers_file:
        for raw_line in papers_file:
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            citation_ids = record.get("citation_ids", []) or []
            citation_id_set = set(citation_ids)
            if len(citation_id_set) > CITATION_CAP:
                raise ValueError(
                    f"[{condition}] node {node} paper {record.get('id')} has "
                    f"{len(citation_id_set)} unique citation_ids (cap={CITATION_CAP})"
                )
            sets.append(citation_id_set)

    if len(sets) != PAPERS_PER_NODE:
        raise ValueError(
            f"[{condition}] node {node} has {len(sets)} paper records in "
            f"{path.name}, expected {PAPERS_PER_NODE}"
        )
    return sets


def mean_raw_overlap(sets: List[Set[str]]) -> Tuple[float, int]:
    total = 0.0
    n_pairs = 0
    for a, b in combinations(sets, 2):
        total += len(a & b)
        n_pairs += 1
    if n_pairs == 0:
        return 0.0, 0
    return total / n_pairs, n_pairs


def analyze_concentration_for_node(
    condition: str, node: int, valid_ids: Set[str]
) -> Dict:
    stats, stray_count = load_node_stats(condition, node, valid_ids)
    if stray_count > 0:
        print(
            f"[{condition}] Node {node}: WARNING {stray_count} stray citation "
            f"id(s) in stats file not in valid_ids; filtered out"
        )
    sorted_stats = sorted(stats.items(), key=_sort_key_count_desc)
    total_citations = sum(stats.values())
    available = available_papers(node)

    frac_10, k_10 = top_percent_share(
        sorted_stats, TOP_PERCENT, total_citations, available
    )

    return {
        "condition": condition,
        "node": node,
        "total_citations": total_citations,
        "unique_papers_cited": len(stats),
        "available_papers": available,
        "top_10pct_share": round(100.0 * frac_10, 2),
        "top_10pct_count": k_10,
    }


def analyze_overlap_for_node(condition: str, node: int) -> Dict:
    sets = load_node_citation_sets(condition, node)
    mean_overlap, n_pairs_used = mean_raw_overlap(sets)
    return {
        "condition": condition,
        "node": node,
        "n_papers": len(sets),
        "n_pairs": n_pairs_used,
        "mean_raw_overlap": round(mean_overlap, 6),
    }


def run_condition(
    condition: str, valid_ids: Set[str]
) -> Tuple[List[Dict], List[Dict]]:
    concentration_rows: List[Dict] = []
    overlap_rows: List[Dict] = []
    for node in range(TOTAL_NODES):
        conc = analyze_concentration_for_node(condition, node, valid_ids)
        overlap = analyze_overlap_for_node(condition, node)
        concentration_rows.append(conc)
        overlap_rows.append(overlap)
        print(
            f"[{condition}] Node {node}: total={conc['total_citations']}, "
            f"top10pct={conc['top_10pct_share']}% "
            f"(n={conc['top_10pct_count']}), "
            f"mean_raw_overlap={overlap['mean_raw_overlap']:.6f} "
            f"(pairs={overlap['n_pairs']})"
        )
    return concentration_rows, overlap_rows



def write_jsonl(rows_by_condition: Dict[str, List[Dict]], out_path: Path) -> None:
    with out_path.open("w") as out_file:
        for condition in CONDITIONS:
            for row in rows_by_condition[condition]:
                out_file.write(json.dumps(row) + "\n")
    print(f"Saved {out_path}")


def plot_concentration_and_overlap(
    concentration_rows_by_condition: Dict[str, List[Dict]],
    overlap_rows_by_condition: Dict[str, List[Dict]],
    out_path: Path,
) -> None:
    fig, (ax_conc, ax_jac) = plt.subplots(
        1,
        2,
        figsize=(10.0, 3.8),
        sharex=True,
        constrained_layout=False,
    )

    x = np.arange(TOTAL_NODES)
    x_labels = [str(n) for n in range(TOTAL_NODES)]

    all_conc: List[float] = []
    all_overlap: List[float] = []

    for condition in CONDITIONS:
        conc_rows = concentration_rows_by_condition[condition]
        overlap_rows = overlap_rows_by_condition[condition]
        y_conc = [r["top_10pct_share"] for r in conc_rows]
        y_overlap = [r["mean_raw_overlap"] for r in overlap_rows]
        all_conc.extend(y_conc)
        all_overlap.extend(y_overlap)
        ax_conc.plot(
            x,
            y_conc,
            color=CONDITION_COLORS[condition],
            linestyle=CONDITION_LINESTYLE[condition],
            marker="o",
            linewidth=2,
            label=CONDITION_LEGEND[condition],
        )
        ax_jac.plot(
            x,
            y_overlap,
            color=CONDITION_COLORS[condition],
            linestyle=CONDITION_LINESTYLE[condition],
            marker="o",
            linewidth=2,
            label=CONDITION_LEGEND[condition],
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=CONDITION_COLORS[c],
            linestyle=CONDITION_LINESTYLE[c],
            marker="o",
            markersize=5.5,
            linewidth=2,
            label=CONDITION_LEGEND[c],
        )
        for c in CONDITIONS
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
        handlelength=3.8,
    )

    ax_conc.set_ylabel("Top 10% Share (%)")
    finite_conc = [v for v in all_conc if np.isfinite(v)]
    y_top_conc = min(100, max(finite_conc) * 1.12) if finite_conc else 100
    ax_conc.set_ylim(0, y_top_conc)
    # Multiples of 10 read naturally for percentages (MaxNLocator alone often picks e.g. 8).
    ax_conc.yaxis.set_major_locator(MultipleLocator(10))
    ax_conc.text(
        0.5,
        1.04,
        "(a)",
        transform=ax_conc.transAxes,
        fontsize=11,
        va="bottom",
        ha="center",
    )

    for ax in (ax_conc, ax_jac):
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("Node")

    ax_jac.set_ylabel("Mean Pairwise Overlap")
    finite_overlap = [v for v in all_overlap if np.isfinite(v)]
    y_top_overlap = max(finite_overlap) * 1.12 if finite_overlap else 1.0
    ax_jac.set_ylim(0, y_top_overlap)
    ax_jac.yaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=5))
    ax_jac.text(
        0.5,
        1.04,
        "(b)",
        transform=ax_jac.transAxes,
        fontsize=11,
        va="bottom",
        ha="center",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


def main() -> None:
    valid_ids = load_valid_ids()
    print(f"Loaded {len(valid_ids)} valid ids from {PRESAMPLED_KV_PATH}")

    concentration_rows_by_condition: Dict[str, List[Dict]] = {}
    overlap_rows_by_condition: Dict[str, List[Dict]] = {}
    for condition in CONDITIONS:
        print(f"\n=== {condition} ===")
        conc_rows, overlap_rows = run_condition(condition, valid_ids)
        concentration_rows_by_condition[condition] = conc_rows
        overlap_rows_by_condition[condition] = overlap_rows

    write_jsonl(concentration_rows_by_condition, CONCENTRATION_OUTPUT_PATH)
    write_jsonl(overlap_rows_by_condition, OVERLAP_OUTPUT_PATH)

    plot_concentration_and_overlap(
        concentration_rows_by_condition,
        overlap_rows_by_condition,
        COMBINED_FIGURE_PATH,
    )


if __name__ == "__main__":
    main()
