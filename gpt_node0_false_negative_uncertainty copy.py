import json, os, re, math
from collections import defaultdict
import numpy as np

def load_valid_ids(kv_pairs_jsonl: str) -> set[str]:
    valid = set()
    with open(kv_pairs_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                valid.add(json.loads(line)["id"])
    return valid

def list_run_dirs(random_root: str) -> list[str]:
    pat = re.compile(r"^run_(\d+)$")
    runs = []
    for name in os.listdir(random_root):
        if pat.match(name):
            p = os.path.join(random_root, name)
            if os.path.isdir(p):
                runs.append(p)
    return sorted(runs)

def build_entries_for_node(run_dir: str, node: int, valid_ids: set[str]):
    citations = defaultdict(int)
    exposures = defaultdict(int)

    stats_path = os.path.join(run_dir, f"node_{node}", f"node_{node}_stats.jsonl")
    with open(stats_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            for pid, cnt in d.items():
                if pid in valid_ids:
                    citations[pid] += int(cnt)

    node_path = os.path.join(run_dir, f"node_{node}", f"node_{node}.jsonl")
    with open(node_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            paper = json.loads(line)
            for pid in paper.get("papers_seen_id", []):
                if pid in valid_ids:
                    exposures[pid] += 1

    all_ids = set(citations) | set(exposures)
    return [{"id": pid, "citations": citations[pid], "exposures": exposures[pid]} for pid in all_ids]

def shown_ignored_rate_of_shown(entries) -> float:
    cited = sum(1 for e in entries if e["citations"] > 0)
    shown_ignored = sum(1 for e in entries if e["exposures"] > 0 and e["citations"] == 0)
    shown_papers = cited + shown_ignored
    return (shown_ignored / shown_papers) if shown_papers > 0 else 0.0

def summarize(values):
    x = np.asarray(values, dtype=float)
    n = x.size
    mean = float(x.mean()) if n else float("nan")
    std = float(x.std(ddof=1)) if n >= 2 else 0.0
    sem = std / math.sqrt(n) if n >= 2 else 0.0
    ci95 = (mean - 1.96 * sem, mean + 1.96 * sem)
    return {"n": int(n), "mean": mean, "std": std, "sem": sem, "ci95": ci95}

if __name__ == "__main__":
    kv = "gpt/output/master/kv_pairs.jsonl"
    random_root = "gpt/random_output"
    node = 0

    valid_ids = load_valid_ids(kv)
    vals = []
    for run_dir in list_run_dirs(random_root):
        entries = build_entries_for_node(run_dir, node, valid_ids)
        vals.append(shown_ignored_rate_of_shown(entries))

    print("GPT random baseline Node 0 shown_ignored_rate_of_shown:", summarize(vals))