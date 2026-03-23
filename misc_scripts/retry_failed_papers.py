"""
Retry failed API calls for N3P82 and N9P23, then update all relevant output files.

Files updated per node:
  - output/node_{n}/node_{n}.jsonl          (append new paper)
  - output/node_{n}/node_{n}_stats.jsonl    (rewrite with updated citation counts)
  - output/node_{n}/node_{n}_token_usage.jsonl  (append token record)
  - output/node_{n}/node_{n}_token_totals.jsonl (rewrite with updated totals)

Global files updated:
  - output/citation_counts/citation_counts.jsonl  (rewrite with updated counts)
  - output/citation_counts/hallucinations.jsonl   (append if any)
  - output/citation_counts/over_cap_violations.jsonl (append if any)

After running this script, re-run false_negatives.py to regenerate
exposure files and exclusion_rate figures.
"""

import asyncio
import json
import random
import sys
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from citation_parser import extract_citations_from_body

load_dotenv()

CITATION_CAP = 10
MODEL = "gpt-5-mini"
SEED = 42

client = AsyncOpenAI()

# Papers to retry: (paper_id, node)
RETRY_PAPERS = [
    ("N3P82", 3),
    ("N9P23", 9),
]


def load_kv_pairs() -> dict:
    seen = {}
    with open("output/master/kv_pairs.jsonl") as f:
        for line in f:
            data = json.loads(line.strip())
            seen[(data["author"], data["year"])] = data["id"]
    return seen


def get_prompt_data(paper_id: str, node: int) -> dict | None:
    with open(f"prompts/N_{node}_inputs.jsonl") as f:
        for line in f:
            data = json.loads(line.strip())
            if data["id"] == paper_id:
                return data
    return None


async def call_api(prompt: list, paper_id: str) -> tuple[str | None, dict]:
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=prompt,
            max_completion_tokens=5000,
            response_format={"type": "json_object"},
        )
        if response.usage:
            usage["prompt_tokens"] = response.usage.prompt_tokens
            usage["completion_tokens"] = response.usage.completion_tokens
            usage["total_tokens"] = response.usage.total_tokens

        content = response.choices[0].message.content
        if not content:
            print(f"[{paper_id}] Empty response, retrying with higher token limit...")
            await asyncio.sleep(5)
            response = await client.chat.completions.create(
                model=MODEL,
                messages=prompt,
                max_completion_tokens=10000,
                response_format={"type": "json_object"},
            )
            if response.usage:
                usage["prompt_tokens"] += response.usage.prompt_tokens
                usage["completion_tokens"] += response.usage.completion_tokens
                usage["total_tokens"] += response.usage.total_tokens
            content = response.choices[0].message.content

        print(
            f"[{paper_id}] Input: {usage['prompt_tokens']} tokens | "
            f"Output: {usage['completion_tokens']} tokens | "
            f"Total: {usage['total_tokens']} tokens"
        )
        return content, usage

    except Exception as e:
        print(f"[{paper_id}] API Error: {e}")
        await asyncio.sleep(5)
        return None, usage


async def retry_paper(paper_id: str, node: int, seen_author_year_pairs: dict) -> dict | None:
    prompt_data = get_prompt_data(paper_id, node)
    if not prompt_data:
        print(f"Could not find prompt for {paper_id} in node {node} prompt file.")
        return None

    content, usage = await call_api(prompt_data["prompt"], paper_id)
    if not content:
        print(f"[{paper_id}] Failed to get a response.")
        return None

    json_response = json.loads(content)
    title = json_response.get("title", "").strip()
    abstract = json_response.get("abstract", "").strip()
    body = json_response.get("body", "").strip()

    raw_citations = extract_citations_from_body(body)
    citation_id_set = set()
    hallucinations = []
    over_cap_violations = []

    for cite in raw_citations:
        if cite in seen_author_year_pairs:
            citation_id_set.add(seen_author_year_pairs[cite])
        else:
            print(f"[{paper_id}] Warning: Potential hallucinated citation - {cite}")
            hallucinations.append({paper_id: list(cite)})

    if len(citation_id_set) > CITATION_CAP:
        print(
            f"[{paper_id}] OVER CAP: {len(citation_id_set)} unique citations, "
            f"truncating to {CITATION_CAP}"
        )
        over_cap_violations.append({
            "paper_id": paper_id,
            "node": node,
            "original_count": len(citation_id_set),
            "original_citation_ids": sorted(citation_id_set),
        })
        citation_id_set = set(random.sample(sorted(citation_id_set), CITATION_CAP))

    output_record = {
        "id": paper_id,
        "author": prompt_data["author"],
        "year": prompt_data["year"],
        "type": prompt_data["type"],
        "title": title,
        "abstract": abstract,
        "body": body,
        "papers_seen_id": prompt_data["papers_seen_id"],
        "citations": [list(c) for c in raw_citations],
        "citation_ids": list(citation_id_set),
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "total_tokens": usage["total_tokens"],
    }

    token_record = {
        "id": paper_id,
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "total_tokens": usage["total_tokens"],
    }

    return {
        "node": node,
        "paper_id": paper_id,
        "record": output_record,
        "token_record": token_record,
        "citation_ids": list(citation_id_set),
        "hallucinations": hallucinations,
        "over_cap_violations": over_cap_violations,
    }


def update_stats(node: int, new_citation_ids: list) -> None:
    stats_path = f"output/node_{node}/node_{node}_stats.jsonl"
    existing = {}
    with open(stats_path) as f:
        for line in f:
            data = json.loads(line.strip())
            for k, v in data.items():
                existing[k] = v
    for cited_id in new_citation_ids:
        existing[cited_id] = existing.get(cited_id, 0) + 1
    with open(stats_path, "w") as f:
        for k, v in existing.items():
            f.write(json.dumps({k: v}) + "\n")
    print(f"  Updated node_{node}_stats.jsonl ({len(new_citation_ids)} new citations)")


def update_token_totals(node: int, token_record: dict) -> None:
    totals_path = f"output/node_{node}/node_{node}_token_totals.jsonl"
    with open(totals_path) as f:
        existing = json.loads(f.read().strip())
    existing["total_prompt_tokens"] += token_record["prompt_tokens"]
    existing["total_completion_tokens"] += token_record["completion_tokens"]
    existing["total_tokens"] += token_record["total_tokens"]
    with open(totals_path, "w") as f:
        f.write(json.dumps(existing) + "\n")
    print(f"  Updated node_{node}_token_totals.jsonl")


def update_citation_counts(all_new_citation_ids: list) -> None:
    cc_path = "output/citation_counts/citation_counts.jsonl"
    existing = {}
    with open(cc_path) as f:
        for line in f:
            data = json.loads(line.strip())
            for k, v in data.items():
                existing[k] = v
    for cited_id in all_new_citation_ids:
        existing[cited_id] = existing.get(cited_id, 0) + 1
    with open(cc_path, "w") as f:
        for k, v in existing.items():
            f.write(json.dumps({k: v}) + "\n")
    print(f"  Updated citation_counts.jsonl ({len(all_new_citation_ids)} new citations)")


async def main():
    random.seed(SEED)
    seen_author_year_pairs = load_kv_pairs()
    print(f"Loaded {len(seen_author_year_pairs)} author-year pairs from kv_pairs.jsonl")

    all_new_citation_ids = []

    for paper_id, node in RETRY_PAPERS:
        print(f"\n{'='*40}")
        print(f"Retrying {paper_id} (node {node})...")
        print(f"{'='*40}")

        result = await retry_paper(paper_id, node, seen_author_year_pairs)
        if not result:
            print(f"FAILED: Could not retrieve {paper_id}. Skipping.")
            continue

        # Append to node output file
        node_path = f"output/node_{node}/node_{node}.jsonl"
        with open(node_path, "a") as f:
            f.write(json.dumps(result["record"]) + "\n")
        print(f"  Appended {paper_id} to node_{node}.jsonl")

        # Append to token_usage file
        token_usage_path = f"output/node_{node}/node_{node}_token_usage.jsonl"
        with open(token_usage_path, "a") as f:
            f.write(json.dumps(result["token_record"]) + "\n")
        print(f"  Appended {paper_id} to node_{node}_token_usage.jsonl")

        # Update stats
        update_stats(node, result["citation_ids"])

        # Update token totals
        update_token_totals(node, result["token_record"])

        # Accumulate for global citation_counts update
        all_new_citation_ids.extend(result["citation_ids"])

        # Append hallucinations if any
        if result["hallucinations"]:
            with open("output/citation_counts/hallucinations.jsonl", "a") as f:
                for item in result["hallucinations"]:
                    f.write(json.dumps(item) + "\n")
            print(f"  Appended {len(result['hallucinations'])} hallucination(s)")

        # Append over-cap violations if any
        if result["over_cap_violations"]:
            with open("output/citation_counts/over_cap_violations.jsonl", "a") as f:
                for item in result["over_cap_violations"]:
                    f.write(json.dumps(item) + "\n")
            print(f"  Appended {len(result['over_cap_violations'])} over-cap violation(s)")

        print(f"  Done with {paper_id}")

    # Update global citation_counts
    print(f"\n{'='*40}")
    update_citation_counts(all_new_citation_ids)

    print(f"\n{'='*40}")
    print("All updates complete.")
    print("Next: run 'python metric_scripts/false_negatives.py'")
    print("to regenerate exposure files and exclusion_rate figures.")
    print(f"{'='*40}")


if __name__ == "__main__":
    asyncio.run(main())
