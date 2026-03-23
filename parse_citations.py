import json
from collections import defaultdict
from pathlib import Path

from citation_parser import extract_citations_from_body

BASE = Path(__file__).resolve().parent
OUTPUT_DIR = BASE / "output"
KV_PAIRS_PATH = OUTPUT_DIR / "master" / "kv_pairs.jsonl"
MAX_NODES = 12


def load_ground_truth(path: Path) -> dict[tuple[str, int], str]:
    """Load (author, year) -> id from kv_pairs.jsonl."""
    gt: dict[tuple[str, int], str] = {}
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            key = (entry["author"], entry["year"])
            gt[key] = entry["id"]
    return gt


def run():
    gt = load_ground_truth(KV_PAIRS_PATH)
    print(f"Ground truth: {len(gt)} (author, year) -> id mappings")

    all_hallucinations: list[dict] = []
    overall_extracted = 0
    overall_valid = 0
    overall_hallucinated = 0

    for node in range(MAX_NODES):
        node_path = OUTPUT_DIR / f"node_{node}" / f"node_{node}.jsonl"
        if not node_path.exists():
            print(f"Skip node {node}: {node_path} not found")
            continue

        node_citations: dict[str, int] = defaultdict(int)
        papers: list[dict] = []

        with open(node_path) as f:
            for line in f:
                paper = json.loads(line)
                body = paper.get("body", "")
                raw_citations = extract_citations_from_body(body)
                citation_id_set: set[str] = set()
                paper_hallucinations: list[list] = []

                for (author, year) in raw_citations:
                    key = (author, year)
                    if key in gt:
                        citation_id_set.add(gt[key])
                    else:
                        paper_hallucinations.append([author, year])

                paper["citations"] = [list(c) for c in sorted(raw_citations)]
                paper["citation_ids"] = sorted(citation_id_set)

                for hall in paper_hallucinations:
                    all_hallucinations.append({paper["id"]: hall})

                for cid in citation_id_set:
                    node_citations[cid] += 1

                papers.append(paper)
                overall_extracted += len(raw_citations)
                overall_valid += len(citation_id_set)
                overall_hallucinated += len(paper_hallucinations)

        with open(node_path, "w") as f:
            for p in papers:
                f.write(json.dumps(p) + "\n")

        stats_path = OUTPUT_DIR / f"node_{node}" / f"node_{node}_stats.jsonl"
        with open(stats_path, "w") as f:
            for paper_id, count in sorted(node_citations.items()):
                f.write(json.dumps({paper_id: count}) + "\n")

        print(f"Node {node}: wrote {node_path} and {stats_path}")

    hall_path = OUTPUT_DIR / "citation_counts" / "hallucinations.jsonl"
    hall_path.parent.mkdir(parents=True, exist_ok=True)
    with open(hall_path, "w") as f:
        for entry in all_hallucinations:
            f.write(json.dumps(entry) + "\n")

    print(f"\nWrote {hall_path} ({len(all_hallucinations)} hallucination entries)")
    print(f"Overall: extracted={overall_extracted}, valid={overall_valid}, hallucinated={overall_hallucinated}")


if __name__ == "__main__":
    run()
