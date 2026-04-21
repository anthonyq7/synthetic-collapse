import json, os, re, math
import numpy as np

SEED_COUNT = 120
PAPERS_PER_NODE = 120

def get_available_papers(node: int) -> int:
    return SEED_COUNT + PAPERS_PER_NODE * node

def load_valid_ids(kv_pairs_jsonl: str) -> set[str]:
    valid = set()
    with open(kv_pairs_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                valid.add(json.loads(line)["id"])
    return valid

def load_node_stats(stats_jsonl: str, valid_ids: set[str]) -> dict[str, int]:
    stats = {}
    with open(stats_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            for k, v in d.items():
                if k in valid_ids:
                    stats[k] = int(v)
    return stats

def top10_share_percent(stats: dict[str, int], node: int) -> float:
    items = sorted(stats.items(), key=lambda x: (-x[1], x[0]))
    total = sum(stats.values())
    if total <= 0:
        return 0.0
    available = get_available_papers(node)
    k = max(1, int(0.10 * available))
    k_eff = min(k, len(items))
    top_cites = sum(c for _, c in items[:k_eff])
    return 100.0 * top_cites / total

def list_run_dirs(random_root: str) -> list[str]:
    pat = re.compile(r"^run_(\d+)$")
    runs = []
    for name in os.listdir(random_root):
        if pat.match(name):
            p = os.path.join(random_root, name)
            if os.path.isdir(p):
                runs.append(p)
    return sorted(runs)

def summarize(values):
    x = np.asarray(values, dtype=float)
    n = x.size
    mean = float(x.mean()) if n else float("nan")
    std = float(x.std(ddof=1)) if n >= 2 else 0.0
    sem = std / math.sqrt(n) if n >= 2 else 0.0
    ci95 = (mean - 1.96 * sem, mean + 1.96 * sem)
    return {"n": int(n), "mean": mean, "std": std, "sem": sem, "ci95": ci95}

if __name__ == "__main__":
    kv = "claude/output/master/kv_pairs.jsonl"
    random_root = "claude/random_output"
    node = 0

    valid_ids = load_valid_ids(kv)
    vals = []
    for run_dir in list_run_dirs(random_root):
        stats_path = os.path.join(run_dir, f"node_{node}", f"node_{node}_stats.jsonl")
        stats = load_node_stats(stats_path, valid_ids)
        vals.append(top10_share_percent(stats, node))

    print("Claude random baseline Node 0 top_10pct_share (%):", summarize(vals))