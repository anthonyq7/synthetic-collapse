import json
from pathlib import Path

TOTAL_NODES = 12
NODE_SIZE = 120
TOTAL_PAPERS = TOTAL_NODES * NODE_SIZE

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parent / "output"


def load_node_token_usage(node: int):
    """Load per-paper token usage for one node."""
    path = OUTPUT_DIR / f"node_{node}" / f"node_{node}_token_usage.jsonl"
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def main():
    # Aggregate from per-paper token_usage.jsonl (already collected by experiment)
    prompt_sum = 0
    completion_sum = 0
    count = 0
    by_node = []
    for node in range(TOTAL_NODES):
        records = load_node_token_usage(node)
        node_prompt = sum(r["prompt_tokens"] for r in records)
        node_completion = sum(r["completion_tokens"] for r in records)
        prompt_sum += node_prompt
        completion_sum += node_completion
        count += len(records)
        by_node.append({
            "node": node,
            "papers": len(records),
            "total_prompt_tokens": node_prompt,
            "total_completion_tokens": node_completion,
            "mean_prompt_tokens_per_paper": round(node_prompt / len(records), 2) if records else 0,
            "mean_completion_tokens_per_paper": round(node_completion / len(records), 2) if records else 0,
        })

    mean_input_tokens = prompt_sum / count if count else 0
    mean_output_tokens = completion_sum / count if count else 0

    estimates = {
        "total_prompt_tokens": prompt_sum,
        "total_completion_tokens": completion_sum,
        "total_tokens": prompt_sum + completion_sum,
        "total_papers": count,
        "mean_input_tokens_per_paper": round(mean_input_tokens, 3),
        "mean_output_tokens_per_paper": round(mean_output_tokens, 3),
        "by_node": by_node,
    }

    output_path = OUTPUT_DIR / "token_estimates.json"
    with open(output_path, "w") as f:
        json.dump(estimates, f, indent=2)

    print(f"Saved token estimates to {output_path}")
    print(f"  total_prompt_tokens: {estimates['total_prompt_tokens']}")
    print(f"  total_completion_tokens: {estimates['total_completion_tokens']}")
    print(f"  total_tokens: {estimates['total_tokens']}")
    print(f"  total_papers: {estimates['total_papers']}")
    print(f"  mean_input_tokens_per_paper: {estimates['mean_input_tokens_per_paper']}")
    print(f"  mean_output_tokens_per_paper: {estimates['mean_output_tokens_per_paper']}")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Ensure the experiment has been run and output files exist.")
        raise
