import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

BASE = "."
MAX_NODES = 12
OUTPUT_ROOT = f"{BASE}/output"


def load_exposure(node: int, data_root: str) -> list[dict]:
    path = f"{data_root}/node_{node}/node_{node}_exposure.jsonl"
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_rates(entries: list[dict]) -> dict:
    seed_citations = seed_exposures = 0
    llm_citations = llm_exposures = 0

    for e in entries:
        pid = e["id"]
        c, ex = e["citations"], e["exposures"]
        if pid.startswith("SEED_"):
            seed_citations += c
            seed_exposures += ex
        else:
            llm_citations += c
            llm_exposures += ex

    seed_rate = (
        seed_citations / seed_exposures if seed_exposures > 0 else None
    )
    llm_rate = llm_citations / llm_exposures if llm_exposures > 0 else None

    return {
        "seed_citations": seed_citations,
        "seed_exposures": seed_exposures,
        "seed_rate": seed_rate,
        "llm_citations": llm_citations,
        "llm_exposures": llm_exposures,
        "llm_rate": llm_rate,
    }


def rates_for_node(node: int, data_root: str) -> dict:
    entries = load_exposure(node, data_root)
    return compute_rates(entries)


def main():
    results = []
    for node in range(MAX_NODES):
        r = rates_for_node(node, OUTPUT_ROOT)
        results.append({
            "node": node,
            "seed_citations": r["seed_citations"],
            "seed_exposures": r["seed_exposures"],
            "seed_rate": round(r["seed_rate"], 6) if r["seed_rate"] is not None else None,
            "llm_citations": r["llm_citations"],
            "llm_exposures": r["llm_exposures"],
            "llm_rate": round(r["llm_rate"], 6) if r["llm_rate"] is not None else None,
        })

    print(f"{'Node':>4}  {'Seed Rate':>10}  {'Seed C/E':>14}  {'LLM Rate':>10}  {'LLM C/E':>14}")
    print("-" * 60)
    for r in results:
        seed_str = f"{r['seed_citations']}/{r['seed_exposures']}"
        llm_str = f"{r['llm_citations']}/{r['llm_exposures']}"
        seed_rate = f"{r['seed_rate']:.4f}" if r["seed_rate"] is not None else "N/A"
        llm_rate = f"{r['llm_rate']:.4f}" if r["llm_rate"] is not None else "N/A"
        print(
            f"{r['node']:>4}  {seed_rate:>10}  {seed_str:>14}  {llm_rate:>10}  {llm_str:>14}"
        )

    out_path = f"{OUTPUT_ROOT}/master/seed_vs_llm_citation_rate.jsonl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nSaved {out_path}")

    nodes = [r["node"] for r in results]
    seed_rates = [r["seed_rate"] for r in results]
    llm_rates = [r["llm_rate"] if r["llm_rate"] is not None else 0 for r in results]

    x = np.arange(len(nodes))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        x,
        seed_rates,
        color="#E74C3C",
        marker="o",
        linewidth=2,
        label="LLM citer — seed papers",
    )
    ax.plot(
        x,
        llm_rates,
        color="#3498DB",
        marker="s", 
        linewidth=2,
        linestyle="--",
        label="LLM citer — LLM papers",
    )

    ax.set_xlabel("Node (Generation)")
    ax.set_ylabel("Citation Rate (citations / exposures) %")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Node {n}" for n in nodes])
    ax.set_ylim(0, None)
    ax.legend(loc="best", fontsize=8)

    y_lo, y_hi = ax.get_ylim()
    y_range = y_hi - y_lo
    for i, v in enumerate(seed_rates):
        if v is not None:
            offset = (0, -14) if (y_hi - v) < 0.08 * y_range else (0, 8)
            ax.annotate(f"{v * 100:.1f}%", (x[i], v), textcoords="offset points",
                        xytext=offset, ha="center", fontsize=8, color="#E74C3C")
    for i, v in enumerate(llm_rates):
        offset = (0, 8) if v < 0.08 * y_range else (0, -14)
        ax.annotate(f"{v * 100:.1f}%", (x[i], v), textcoords="offset points",
                    xytext=offset, ha="center", fontsize=8, color="#3498DB")
    fig.suptitle(
        "Figure 3: Citation Rate of Seed vs LLM-Generated Papers",
        fontsize=12,
    )
    fig.tight_layout()

    fig_path = f"{OUTPUT_ROOT}/master/seed_vs_llm_citation_rate.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
