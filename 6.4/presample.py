import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List

from faker import Faker


NODE_SIZE = 120
STRATUM_SIZE = 60
TOTAL_NODES = 12
PAPER_SET_LENGTH = 30
SEED = 42

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SEED_PATH = REPO_ROOT / "gpt" / "output" / "seed" / "seed.jsonl"
PRESAMPLED_DIR = SCRIPT_DIR / "presampled"

MODEL_L1_TAG = "gpt"
MODEL_L2_COUNTS = {"gpt": 60, "haiku": 60}
MODEL_L3_COUNTS = {"gpt": 40, "haiku": 40, "gemini": 40}

fake = Faker()
Faker.seed(SEED)
_SIMPLE_NAME = re.compile(r"^[A-Z][a-z]+$")


def simple_surname() -> str:
    name = fake.last_name()
    while not _SIMPLE_NAME.match(name):
        name = fake.last_name()
    return name


def load_seed_papers() -> List[Dict]:
    if not SEED_PATH.exists():
        raise FileNotFoundError(
            f"Expected seed file at {SEED_PATH}. Run gpt/experiment.py's "
            f"standardize_seed() first."
        )
    seed_papers: List[Dict] = []
    with SEED_PATH.open("r") as seed_file:
        for raw_line in seed_file:
            line = raw_line.strip()
            if not line:
                continue
            paper = json.loads(line)
            seed_papers.append(
                {
                    "id": paper["id"],
                    "author": paper["author"],
                    "year": paper["year"],
                    "title": paper.get("title", ""),
                    "abstract": paper.get("abstract", ""),
                }
            )
    return seed_papers


def build_model_list_for_node(counts: Dict[str, int]) -> List[str]:
    model_list: List[str] = []
    for model_tag, count in counts.items():
        for _ in range(count):
            model_list.append(model_tag)
    if len(model_list) != NODE_SIZE:
        raise ValueError(
            f"Model counts {counts} sum to {len(model_list)} but NODE_SIZE={NODE_SIZE}"
        )
    random.shuffle(model_list)
    return model_list


def build_stratified_model_list_for_node(pos_indices: set) -> List[str]:
    """Assign 30 gpt + 30 haiku to position papers and 30 gpt + 30 haiku to lit reviews."""
    pos_models = ["gpt"] * 30 + ["haiku"] * 30
    lit_models = ["gpt"] * 30 + ["haiku"] * 30
    random.shuffle(pos_models)
    random.shuffle(lit_models)

    pos_iter = iter(pos_models)
    lit_iter = iter(lit_models)

    assignments: List[str] = []
    for i in range(NODE_SIZE):
        if i in pos_indices:
            assignments.append(next(pos_iter))
        else:
            assignments.append(next(lit_iter))
    return assignments


def presample_all_nodes() -> None:
    random.seed(SEED)

    PRESAMPLED_DIR.mkdir(parents=True, exist_ok=True)

    seed_papers = load_seed_papers()
    if len(seed_papers) != NODE_SIZE:
        print(
            f"Warning: seed has {len(seed_papers)} papers but NODE_SIZE={NODE_SIZE}; "
            f"continuing with available seed size."
        )

    seen_author_year_pairs: Dict[tuple, str] = {}
    for seed_paper in seed_papers:
        pair_key = (seed_paper["author"], seed_paper["year"])
        seen_author_year_pairs[pair_key] = seed_paper["id"]

    pool_records: List[Dict] = []
    for seed_paper in seed_papers:
        pool_records.append(
            {
                "id": seed_paper["id"],
                "author": seed_paper["author"],
                "year": seed_paper["year"],
                "title": seed_paper["title"],
                "abstract": seed_paper["abstract"],
            }
        )

    for node in range(TOTAL_NODES):
        print(f"Presampling node {node}...")

        pos_indices = set(random.sample(range(NODE_SIZE - 1), STRATUM_SIZE))
        node_records: List[Dict] = []
        new_pool_additions: List[Dict] = []

        for paper_index in range(NODE_SIZE):
            paper_id = f"N{node}P{paper_index}"

            candidate_set = random.sample(pool_records, PAPER_SET_LENGTH)
            candidate_set_snapshot = []
            for candidate in candidate_set:
                candidate_set_snapshot.append(
                    {
                        "id": candidate["id"],
                        "author": candidate["author"],
                        "year": candidate["year"],
                        "title": candidate["title"],
                        "abstract": candidate["abstract"],
                    }
                )

            author = simple_surname()
            year = random.randint(2017, 2025)
            while (author, year) in seen_author_year_pairs:
                author = simple_surname()
                year = random.randint(2017, 2025)
            seen_author_year_pairs[(author, year)] = paper_id

            if paper_index in pos_indices:
                paper_type = "position paper"
            else:
                paper_type = "literature review"

            node_records.append(
                {
                    "id": paper_id,
                    "author": author,
                    "year": year,
                    "type": paper_type,
                    "candidate_set": candidate_set_snapshot,
                }
            )

            new_pool_additions.append(
                {
                    "id": paper_id,
                    "author": author,
                    "year": year,
                    "title": "",
                    "abstract": "",
                }
            )

        pool_records.extend(new_pool_additions)

        model_L1_assignments = [MODEL_L1_TAG] * NODE_SIZE
        model_L2_assignments = build_stratified_model_list_for_node(pos_indices)
        model_L3_assignments = build_model_list_for_node(MODEL_L3_COUNTS)

        output_path = PRESAMPLED_DIR / f"node_{node}_meta.jsonl"
        with output_path.open("w") as output_file:
            for record_index, record in enumerate(node_records):
                record_with_models = {
                    "id": record["id"],
                    "author": record["author"],
                    "year": record["year"],
                    "type": record["type"],
                    "candidate_set": record["candidate_set"],
                    "model_L1": model_L1_assignments[record_index],
                    "model_L2": model_L2_assignments[record_index],
                    "model_L3": model_L3_assignments[record_index],
                    "model": "",
                }
                output_file.write(json.dumps(record_with_models) + "\n")

        print(f"  Wrote {len(node_records)} records to {output_path}")

    kv_pairs_path = PRESAMPLED_DIR / "kv_pairs.jsonl"
    with kv_pairs_path.open("w") as kv_file:
        for (author, year), paper_id in seen_author_year_pairs.items():
            kv_file.write(
                json.dumps({"author": author, "year": year, "id": paper_id}) + "\n"
            )
    print(f"Wrote {len(seen_author_year_pairs)} kv pairs to {kv_pairs_path}")


if __name__ == "__main__":
    presample_all_nodes()
