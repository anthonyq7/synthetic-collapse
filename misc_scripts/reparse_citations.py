"""
Re-parses all LLM outputs across nodes and rewrites citation_ids using
first-10 truncation (order of appearance in body text) instead of random sampling.

Rewrites:
  - output/node_N/node_N.jsonl          (citation_ids field)
  - output/node_N/node_N_stats.jsonl    (citation counts per paper)
  - output/citation_counts/hallucinations.jsonl
  - output/citation_counts/over_cap_violations.jsonl
"""

import json
import re
from collections import defaultdict
from pathlib import Path

BASE = Path(".")
OUTPUT_DIR = BASE / "output"
KV_PAIRS_PATH = OUTPUT_DIR / "master" / "kv_pairs.jsonl"
MAX_NODES = 12
CITATION_CAP = 10

_CITE_PATTERN = re.compile(
    r"\[?\s*([A-Z][a-z]+)\s*\]?\s*,\s*\[?\s*(\d{4})\s*\]?"
)


def load_ground_truth(path: Path) -> dict[tuple[str, int], str]:
    gt = {}
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            gt[(entry["author"], entry["year"])] = entry["id"]
    return gt


def extract_citations_ordered(text: str) -> list[tuple[str, int]]:
    """
    Extract (author, year) citations in order of first appearance in body text.
    Deduplicates: only the first occurrence of each unique (author, year) is kept.
    """
    seen = set()
    ordered = []
    for group in re.finditer(r"\(([^)]+)\)", text):
        for segment in group.group(1).split(";"):
            for m in _CITE_PATTERN.finditer(segment):
                key = (m.group(1).strip(), int(m.group(2)))
                if key not in seen:
                    seen.add(key)
                    ordered.append(key)
    return ordered


def main():
    gt = load_ground_truth(KV_PAIRS_PATH)
    print(f"Loaded {len(gt)} ground truth (author, year) -> id mappings")

    all_hallucinations = []
    all_over_cap_violations = []

    for node in range(MAX_NODES):
        node_path = OUTPUT_DIR / f"node_{node}" / f"node_{node}.jsonl"
        node_citations = defaultdict(int)
        updated_papers = []

        with open(node_path) as f:
            for line in f:
                paper = json.loads(line)
                body = paper.get("body", "")

                ordered_raw = extract_citations_ordered(body)

                # Resolve to IDs in order, deduplicate IDs (different author-year
                # could theoretically map to the same ID, though unlikely)
                citation_ids_ordered = []
                seen_ids = set()
                for key in ordered_raw:
                    if key in gt:
                        cid = gt[key]
                        if cid not in seen_ids:
                            seen_ids.add(cid)
                            citation_ids_ordered.append(cid)
                    else:
                        all_hallucinations.append({paper["id"]: list(key)})

                # Over-cap: take first CITATION_CAP instead of random sample
                if len(citation_ids_ordered) > CITATION_CAP:
                    all_over_cap_violations.append({
                        "paper_id": paper["id"],
                        "node": node,
                        "original_count": len(citation_ids_ordered),
                        "original_citation_ids": citation_ids_ordered,
                    })
                    citation_ids_ordered = citation_ids_ordered[:CITATION_CAP]

                paper["citation_ids"] = citation_ids_ordered
                updated_papers.append(paper)

                for cid in citation_ids_ordered:
                    node_citations[cid] += 1

        with open(node_path, "w") as f:
            for p in updated_papers:
                f.write(json.dumps(p) + "\n")

        stats_path = OUTPUT_DIR / f"node_{node}" / f"node_{node}_stats.jsonl"
        with open(stats_path, "w") as f:
            for paper_id, count in sorted(node_citations.items()):
                f.write(json.dumps({paper_id: count}) + "\n")

        print(f"Node {node}: {len(updated_papers)} papers, "
              f"{sum(node_citations.values())} total citations, "
              f"{len(all_over_cap_violations)} over-cap violations so far")

    hall_path = OUTPUT_DIR / "citation_counts" / "hallucinations.jsonl"
    with open(hall_path, "w") as f:
        for entry in all_hallucinations:
            f.write(json.dumps(entry) + "\n")

    ocv_path = OUTPUT_DIR / "citation_counts" / "over_cap_violations.jsonl"
    with open(ocv_path, "w") as f:
        for entry in all_over_cap_violations:
            f.write(json.dumps(entry) + "\n")

    print(f"\nTotal hallucinations: {len(all_hallucinations)}")
    print(f"Total over-cap violations: {len(all_over_cap_violations)}")
    print(f"Saved {hall_path}")
    print(f"Saved {ocv_path}")


if __name__ == "__main__":
    main()
