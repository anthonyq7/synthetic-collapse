import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

# --- Configuration ---
MODELS = ["gpt", "claude"]
OUT_DIRNAME = "figures"
OUT_FILENAME = "multimodel_seed_catalog_vs_attention_share.png"


# --- Helpers ---
def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def unique_ids_from_jsonl(path: Path, id_key: str) -> set[str]:
    ids: set[str] = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if id_key not in obj:
                raise KeyError(f"Expected key '{id_key}' in {path}, got keys={list(obj.keys())}")
            ids.add(str(obj[id_key]))
    return ids


def infer_repo_root() -> Path:
    """
    Robustly infer the repository root, regardless of where the script is invoked from.

    Priority:
      1) If this file is in the repo root, use its parent.
      2) Otherwise, walk upwards until we find both 'gpt' and 'claude' directories.
    """
    here = Path(__file__).resolve()
    # case (1)
    candidate = here.parent
    if (candidate / "gpt").is_dir() and (candidate / "claude").is_dir():
        return candidate

    # case (2)
    for parent in [here.parent] + list(here.parents):
        if (parent / "gpt").is_dir() and (parent / "claude").is_dir():
            return parent

    raise RuntimeError(
        "Could not infer repo root. Expected to find 'gpt/' and 'claude/' directories above this script."
    )


def seed_set_size(model_root: Path) -> int:
    """
    Seed set size used for "catalog share".

    Tries (in order):
      1) output/seed/seed.jsonl
      2) output/seed/seed_initial.jsonl
      3) output/master/kv_pairs.jsonl filtered to SEED_* ids

    Counts UNIQUE SEED_* ids to avoid duplicates and to handle kv_pairs that
    contains non-seed (generated) ids too.
    """
    seed_json = model_root / "output" / "seed" / "seed.jsonl"
    seed_initial = model_root / "output" / "seed" / "seed_initial.jsonl"
    kv_pairs = model_root / "output" / "master" / "kv_pairs.jsonl"

    if seed_json.exists():
        return len(unique_ids_from_jsonl(seed_json, "id"))

    if seed_initial.exists():
        return len(unique_ids_from_jsonl(seed_initial, "id"))

    if kv_pairs.exists():
        ids = unique_ids_from_jsonl(kv_pairs, "id")
        seed_ids = {i for i in ids if str(i).startswith("SEED_")}
        if not seed_ids:
            raise ValueError(f"Found kv_pairs.jsonl at {kv_pairs} but no SEED_* ids inside it.")
        return len(seed_ids)

    raise FileNotFoundError(
        f"No seed.jsonl, seed_initial.jsonl, or master/kv_pairs.jsonl found under: {model_root / 'output'}"
    )


@dataclass(frozen=True)
class Series:
    nodes: list[int]
    seed_catalog_share: list[float]
    gen_catalog_share: list[float]
    seed_attention_share: list[float]
    gen_attention_share: list[float]
    seed_available: int


def load_series(repo_root: Path, model: str) -> Series:
    model_root = repo_root / model
    master = model_root / "output" / "master"

    rates_path = master / "seed_vs_llm_citation_rate.jsonl"
    conc_path = master / "concentration.jsonl"

    rates = read_jsonl(rates_path)
    conc = read_jsonl(conc_path)

    rates_by_node = {int(r["node"]): r for r in rates}
    conc_by_node = {int(r["node"]): r for r in conc}

    nodes = sorted(set(rates_by_node) & set(conc_by_node))
    if not nodes:
        raise ValueError(f"No overlapping nodes between {rates_path} and {conc_path}")

    seed_available = seed_set_size(model_root)

    total_available = [int(conc_by_node[n]["available_papers"]) for n in nodes]
    if any(t <= 0 for t in total_available):
        raise ValueError(f"Non-positive available_papers in {conc_path}: {total_available}")

    seed_catalog_share = [seed_available / tot for tot in total_available]
    gen_catalog_share = [1.0 - s for s in seed_catalog_share]

    # Attention share: fraction of all citations that go to seeds at each node
    seed_cit = [int(rates_by_node[n]["seed_citations"]) for n in nodes]
    llm_cit = [int(rates_by_node[n]["llm_citations"]) for n in nodes]
    tot_cit = [a + b for a, b in zip(seed_cit, llm_cit)]
    seed_attention_share = [(a / t) if t else 0.0 for a, t in zip(seed_cit, tot_cit)]
    gen_attention_share = [1.0 - s for s in seed_attention_share]

    # Sanity: shares should be within [0,1] (with small float wiggle)
    def _check_shares(name: str, vals: list[float]):
        bad = [v for v in vals if v < -1e-9 or v > 1 + 1e-9]
        if bad:
            raise ValueError(
                f"{model}: {name} has values outside [0,1]. "
                f"seed_available={seed_available}, example_bad={bad[:3]}"
            )

    _check_shares("seed_catalog_share", seed_catalog_share)
    _check_shares("seed_attention_share", seed_attention_share)

    return Series(
        nodes=nodes,
        seed_catalog_share=seed_catalog_share,
        gen_catalog_share=gen_catalog_share,
        seed_attention_share=seed_attention_share,
        gen_attention_share=gen_attention_share,
        seed_available=seed_available,
    )


def plot():
    repo_root = infer_repo_root()
    # Guard against the exact issue you saw:
    assert (repo_root / "gpt").is_dir() and (repo_root / "claude").is_dir(), f"Bad repo_root: {repo_root}"

    series = {m: load_series(repo_root, m) for m in MODELS}

    fig, axes = plt.subplots(
        nrows=len(MODELS),
        ncols=2,
        figsize=(12, 7),
        sharex=True,
        sharey="col",
    )

    # Normalize axes shape in case someone sets MODELS length to 1 later
    if len(MODELS) == 1:
        axes = [axes]

    for i, model in enumerate(MODELS):
        s = series[model]
        x = s.nodes

        # Catalog share
        ax = axes[i][0]
        ax.plot(x, [v * 100 for v in s.seed_catalog_share], marker="o", label="Seed")
        ax.plot(x, [v * 100 for v in s.gen_catalog_share], marker="s", linestyle="--", label="Generated")
        ax.set_ylim(0, 100)
        ax.set_ylabel(f"{model.upper()}\nShare (%)")
        if i == 0:
            ax.set_title("Catalog share of available papers")
        ax.grid(alpha=0.25)
        if i == 0:
            ax.legend(fontsize=8, loc="upper right")

        # Attention share
        
        ax = axes[i][1]
        ax.plot(x, [v * 100 for v in s.seed_attention_share], marker="o", label="Seed")
        ax.plot(x, [v * 100 for v in s.gen_attention_share], marker="s", linestyle="--", label="Generated")
        # Expected attention under proportional-to-catalog null
        ax.plot(
            x,
            [v * 100 for v in s.seed_catalog_share],
            color="gray",
            linewidth=2,
            linestyle=":",
            label="Expected (∝ catalog)",
        )
        ax.set_ylim(0, 100)
        if i == 0:
            ax.set_title("Attention share of citations")
        ax.grid(alpha=0.25)
        if i == 0:
            ax.legend(fontsize=8, loc="upper right")

    for ax in axes[-1]:
        ax.set_xlabel("Node (generation)")
        ax.set_xticks(series[MODELS[0]].nodes)

    fig.suptitle("Seed set shrinks in catalog but persists in attention (by node)", fontsize=13)
    fig.tight_layout()

    out_dir = repo_root / OUT_DIRNAME
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / OUT_FILENAME
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    plot()