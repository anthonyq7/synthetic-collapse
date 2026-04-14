import json
from pathlib import Path

TOTAL_NODES = 12
NODE_SIZE = 120
TOTAL_PAPERS = TOTAL_NODES * NODE_SIZE
SYSTEM_PROMPT_TOKENS = 172

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


def load_node_papers(node: int) -> dict:
    """Load per-paper papers_seen_id for one node, keyed by paper ID."""
    path = OUTPUT_DIR / f"node_{node}" / f"node_{node}.jsonl"
    papers = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                papers[d["id"]] = d.get("papers_seen_id", [])
    return papers


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

    # --- Per-paper-type token breakdown ---
    # node_0 only sees SEED papers (30 per paper), so:
    #   prompt = SYSTEM_PROMPT_TOKENS + 30 * avg_seed  =>  avg_seed = (prompt - SYSTEM_PROMPT_TOKENS) / 30
    node0_records = load_node_token_usage(0)
    avg_seed_tokens = sum(
        (r["prompt_tokens"] - SYSTEM_PROMPT_TOKENS) / 30 for r in node0_records
    ) / len(node0_records)

    # For nodes 1-11, each paper sees a mix of SEED and LLM papers.
    # avg_llm = Σ(prompt - SYSTEM_PROMPT_TOKENS - n_seed * avg_seed) / Σ(n_llm)  (weighted)
    total_llm_contribution = 0.0
    total_llm_count = 0
    for node in range(1, TOTAL_NODES):
        token_usage = {r["id"]: r["prompt_tokens"] for r in load_node_token_usage(node)}
        papers_seen = load_node_papers(node)
        for paper_id, seen_ids in papers_seen.items():
            pt = token_usage.get(paper_id)
            if pt is None:
                continue
            n_seed = sum(1 for p in seen_ids if p.startswith("SEED"))
            n_llm = sum(1 for p in seen_ids if not p.startswith("SEED"))
            if n_llm > 0:
                total_llm_contribution += pt - SYSTEM_PROMPT_TOKENS - n_seed * avg_seed_tokens
                total_llm_count += n_llm

    avg_llm_tokens = total_llm_contribution / total_llm_count if total_llm_count else 0

    estimates = {
        "total_prompt_tokens": prompt_sum,
        "total_completion_tokens": completion_sum,
        "total_tokens": prompt_sum + completion_sum,
        "total_papers": count,
        "mean_input_tokens_per_paper": round(mean_input_tokens, 3),
        "mean_output_tokens_per_paper": round(mean_output_tokens, 3),
        "system_prompt_tokens": SYSTEM_PROMPT_TOKENS,
        "avg_tokens_per_seed_paper_in_context": round(avg_seed_tokens, 3),
        "avg_tokens_per_llm_paper_in_context": round(avg_llm_tokens, 3),
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
    print(f"  system_prompt_tokens: {estimates['system_prompt_tokens']}")
    print(f"  avg_tokens_per_seed_paper_in_context: {estimates['avg_tokens_per_seed_paper_in_context']}")
    print(f"  avg_tokens_per_llm_paper_in_context: {estimates['avg_tokens_per_llm_paper_in_context']}")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Ensure the experiment has been run and output files exist.")
        raise
