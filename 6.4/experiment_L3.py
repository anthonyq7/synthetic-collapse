import asyncio
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from anthropic import (
    APIConnectionError as AnthropicAPIConnectionError,
    APIError as AnthropicAPIError,
    AsyncAnthropic,
    RateLimitError as AnthropicRateLimitError,
)
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError as GeminiAPIError
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from citation_parser import extract_citations_from_body


load_dotenv()

NODE_SIZE = 120
STRATUM_SIZE = 60
CITATION_CAP = 10
TOTAL_NODES = 12
TARGET_LENGTH = 500
SEED = 42
PAPER_SET_LENGTH = 30
TOPIC = "Knowledge distillation or model compression in deep learning or NLP"

MODEL_GPT = "gpt-5-mini"
MODEL_HAIKU = "claude-haiku-4-5-20251001"
MODEL_GEMINI = "gemini-2.5-flash"
MAX_CONCURRENT_GPT = 120
MAX_CONCURRENT_HAIKU = 30
MAX_CONCURRENT_GEMINI = 5
MAX_NODE_ATTEMPTS = 50

CONDITION = "L3"
CONDITION_DIR = SCRIPT_DIR / CONDITION
PROMPTS_DIR = CONDITION_DIR / "prompts"
OUTPUT_DIR = CONDITION_DIR / "output"
SEED_OUTPUT_DIR = OUTPUT_DIR / "seed"
MASTER_OUTPUT_DIR = OUTPUT_DIR / "master"
CITATION_COUNTS_DIR = OUTPUT_DIR / "citation_counts"
PRESAMPLED_DIR = SCRIPT_DIR / "presampled"
SEED_INPUT_PATH = REPO_ROOT / "gpt" / "output" / "seed" / "seed.jsonl"
KV_PAIRS_PATH = PRESAMPLED_DIR / "kv_pairs.jsonl"

SYSTEM_PROMPT = f"""
    You are a researcher writing about a topic using a provided set of articles.
    Output valid JSON only:
    {{
        "title": "...",
        "abstract": "... no citations ...",
        "body": "... support claims by citing relevant articles inline using parenthetical citation format, e.g. ([Surname], [Year]) ..."
    }}
    Do not include citations in the abstract.
    Only cite articles from the provided list using their exact author and year via inline citations.
    You MUST cite no more than {CITATION_CAP} unique articles in the body.
    """

HAIKU_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "abstract": {"type": "string", "description": "no citations"},
        "body": {
            "type": "string",
            "description": (
                "support claims by citing relevant articles inline using "
                "parenthetical citation format, e.g. ([Surname], [Year])"
            ),
        },
    },
    "required": ["title", "abstract", "body"],
}


class PaperResponse(BaseModel):
    title: str
    abstract: str = Field(description="no citations")
    body: str = Field(
        description=(
            "support claims by citing relevant articles inline using parenthetical "
            "citation format, e.g. ([Surname], [Year])"
        )
    )


GPT_RETRY_EXCEPTIONS = (APIError, APIConnectionError, APITimeoutError, RateLimitError)
HAIKU_RETRY_EXCEPTIONS = (
    AnthropicAPIError,
    AnthropicAPIConnectionError,
    AnthropicRateLimitError,
)
GEMINI_RETRY_EXCEPTIONS = (GeminiAPIError,)


class EmptyResponseError(Exception):
    pass


openai_client = AsyncOpenAI()
anthropic_client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
openai_semaphore = asyncio.Semaphore(MAX_CONCURRENT_GPT)
anthropic_semaphore = asyncio.Semaphore(MAX_CONCURRENT_HAIKU)
gemini_semaphore = asyncio.Semaphore(MAX_CONCURRENT_GEMINI)

CITATION_COUNTS = defaultdict(int)
SEEN_AUTHOR_YEAR_PAIRS: Dict[Tuple[str, int], str] = {}
GENERATED_BY_ID: Dict[str, Dict] = {}


def get_citations(text: str):
    return extract_citations_from_body(text)


def load_seed_records() -> List[Dict]:
    seed_records: List[Dict] = []
    with SEED_INPUT_PATH.open("r") as seed_file:
        for raw_line in seed_file:
            line = raw_line.strip()
            if not line:
                continue
            paper = json.loads(line)
            seed_records.append(
                {
                    "id": paper["id"],
                    "author": paper["author"],
                    "year": paper["year"],
                    "title": paper.get("title", ""),
                    "abstract": paper.get("abstract", ""),
                }
            )
    return seed_records


def load_kv_pairs() -> Dict[Tuple[str, int], str]:
    kv_pairs: Dict[Tuple[str, int], str] = {}
    with KV_PAIRS_PATH.open("r") as kv_file:
        for raw_line in kv_file:
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            kv_pairs[(record["author"], record["year"])] = record["id"]
    return kv_pairs


def hydrate_candidate_set(raw_candidates: List[Dict]) -> List[Dict]:
    hydrated_candidates: List[Dict] = []
    for candidate in raw_candidates:
        candidate_id = candidate["id"]
        has_content = candidate.get("title") and candidate.get("abstract")
        if has_content:
            hydrated_candidates.append(
                {
                    "author": candidate["author"],
                    "year": candidate["year"],
                    "title": candidate["title"],
                    "abstract": candidate["abstract"],
                }
            )
            continue

        generated = GENERATED_BY_ID.get(candidate_id)
        if generated is None:
            raise RuntimeError(
                f"Candidate {candidate_id} missing title/abstract and has not been "
                f"generated yet in this condition. Cannot hydrate."
            )
        hydrated_candidates.append(
            {
                "author": generated["author"],
                "year": generated["year"],
                "title": generated["title"],
                "abstract": generated["abstract"],
            }
        )
    return hydrated_candidates


def build_user_prompt(paper_type: str, hydrated_candidates: List[Dict]) -> str:
    if paper_type == "position paper":
        return f"""
                    Using only the articles provided, argue a position on the following topic.
                    Support claims by citing relevant articles inline.
                    Only cite articles from the provided list using their exact author and year via inline citations.
                    You MUST cite no more than {CITATION_CAP} unique articles in the body.

                    Topic: {TOPIC}

                    Articles:
                    {hydrated_candidates}
                    """
    return f"""
                    Using only the articles provided, synthesize what is known about the following topic.
                    Support claims by citing relevant articles inline.
                    Only cite articles from the provided list using their exact author and year via inline citations.
                    You MUST cite no more than {CITATION_CAP} unique articles in the body.

                    Topic: {TOPIC}

                    Articles:
                    {hydrated_candidates}
                    """


def build_prompt_messages(user_prompt_content: str) -> List[Dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt_content},
    ]


def extract_user_message(prompt_messages: List[Dict]) -> Dict:
    for message in prompt_messages:
        if message.get("role") == "user":
            return {"role": "user", "content": message["content"]}
    raise ValueError("No user message found in prompt_messages")


def extract_user_text(prompt_messages: List[Dict]) -> str:
    for message in prompt_messages:
        if message.get("role") == "user":
            return message["content"]
    raise ValueError("No user message found in prompt_messages")


@retry(
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(10),
    retry=retry_if_exception_type(GPT_RETRY_EXCEPTIONS + (EmptyResponseError,)),
    reraise=True,
)
async def generate_gpt(prompt_messages: List[Dict], paper_id: str, paper_data: Dict):
    async with openai_semaphore:
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        response = await openai_client.chat.completions.create(
            model=MODEL_GPT,
            messages=prompt_messages,
            max_completion_tokens=5000,
            response_format={"type": "json_object"},
        )
        if response.usage:
            usage["prompt_tokens"] += response.usage.prompt_tokens
            usage["completion_tokens"] += response.usage.completion_tokens
            usage["total_tokens"] += response.usage.total_tokens

        content = response.choices[0].message.content
        if not content:
            print(f"Empty GPT response {paper_id}. Retrying with higher token budget...")
            await asyncio.sleep(5)
            response = await openai_client.chat.completions.create(
                model=MODEL_GPT,
                messages=prompt_messages,
                max_completion_tokens=10000,
                response_format={"type": "json_object"},
            )
            if response.usage:
                usage["prompt_tokens"] += response.usage.prompt_tokens
                usage["completion_tokens"] += response.usage.completion_tokens
                usage["total_tokens"] += response.usage.total_tokens

            content = response.choices[0].message.content
            if not content:
                raise EmptyResponseError(
                    f"GPT returned empty content twice for {paper_id}"
                )

        print(
            f"[{paper_id}] (gpt) Input: {usage['prompt_tokens']} tokens | "
            f"Output: {usage['completion_tokens']} tokens | "
            f"Total: {usage['total_tokens']} tokens"
        )

        return paper_id, paper_data, content, usage


def extract_haiku_tool_content(response) -> str:
    for block in response.content:
        if block.type == "tool_use":
            return json.dumps(block.input)
    return ""


@retry(
    wait=wait_exponential(multiplier=1, min=2, max=120),
    stop=stop_after_attempt(10),
    retry=retry_if_exception_type(HAIKU_RETRY_EXCEPTIONS + (EmptyResponseError,)),
    reraise=True,
)
async def generate_haiku(prompt_messages: List[Dict], paper_id: str, paper_data: Dict):
    user_message = extract_user_message(prompt_messages)
    messages_for_haiku = [user_message]
    haiku_tools = [
        {
            "name": "json",
            "description": "Output valid JSON only",
            "input_schema": HAIKU_JSON_SCHEMA,
        }
    ]

    async with anthropic_semaphore:
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        response = await anthropic_client.messages.create(
            model=MODEL_HAIKU,
            system=SYSTEM_PROMPT,
            tools=haiku_tools,
            tool_choice={"type": "tool", "name": "json"},
            messages=messages_for_haiku,
            max_tokens=5000,
        )
        if response.usage:
            usage["prompt_tokens"] += response.usage.input_tokens
            usage["completion_tokens"] += response.usage.output_tokens
            usage["total_tokens"] += (
                response.usage.input_tokens + response.usage.output_tokens
            )

        content = extract_haiku_tool_content(response)
        bad_content = (
            not content
            or content == "{}"
            or response.stop_reason == "max_tokens"
        )
        if bad_content:
            print(f"Empty Haiku response {paper_id}. Retrying with higher token budget...")
            await asyncio.sleep(5)
            response = await anthropic_client.messages.create(
                model=MODEL_HAIKU,
                system=SYSTEM_PROMPT,
                tools=haiku_tools,
                tool_choice={"type": "tool", "name": "json"},
                messages=messages_for_haiku,
                max_tokens=10000,
            )
            if response.usage:
                usage["prompt_tokens"] += response.usage.input_tokens
                usage["completion_tokens"] += response.usage.output_tokens
                usage["total_tokens"] += (
                    response.usage.input_tokens + response.usage.output_tokens
                )

            content = extract_haiku_tool_content(response)
            bad_content = (
                not content
                or content == "{}"
                or response.stop_reason == "max_tokens"
            )
            if bad_content:
                raise EmptyResponseError(
                    f"Haiku returned empty content twice for {paper_id}"
                )

        print(
            f"[{paper_id}] (haiku) Input: {usage['prompt_tokens']} tokens | "
            f"Output: {usage['completion_tokens']} tokens | "
            f"Total: {usage['total_tokens']} tokens"
        )

        return paper_id, paper_data, content, usage


def gemini_usage_delta(response, usage: Dict) -> None:
    if response.usage_metadata is None:
        return
    prompt_tokens = response.usage_metadata.prompt_token_count or 0
    candidates_tokens = response.usage_metadata.candidates_token_count or 0
    thoughts_tokens = response.usage_metadata.thoughts_token_count or 0
    total_tokens = response.usage_metadata.total_token_count or 0
    usage["prompt_tokens"] += prompt_tokens
    usage["completion_tokens"] += candidates_tokens + thoughts_tokens
    usage["total_tokens"] += total_tokens


def gemini_finish_reason_name(response) -> str:
    if not response.candidates:
        return ""
    finish_reason = response.candidates[0].finish_reason
    if finish_reason is None:
        return ""
    return finish_reason.name


@retry(
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(10),
    retry=retry_if_exception_type(GEMINI_RETRY_EXCEPTIONS + (EmptyResponseError,)),
    reraise=True,
)
async def generate_gemini(prompt_messages: List[Dict], paper_id: str, paper_data: Dict):
    user_text = extract_user_text(prompt_messages)

    async with gemini_semaphore:
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        initial_config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=PaperResponse,
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=10000,
        )
        response = await gemini_client.aio.models.generate_content(
            model=MODEL_GEMINI,
            contents=user_text,
            config=initial_config,
        )
        gemini_usage_delta(response, usage)

        content = response.text
        finish_name = gemini_finish_reason_name(response)
        bad_content = (not content) or finish_name == "MAX_TOKENS"

        if bad_content:
            print(f"Empty Gemini response {paper_id}. Retrying with higher token budget...")
            await asyncio.sleep(5)
            retry_config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=PaperResponse,
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=20000,
            )
            response = await gemini_client.aio.models.generate_content(
                model=MODEL_GEMINI,
                contents=user_text,
                config=retry_config,
            )
            gemini_usage_delta(response, usage)

            content = response.text
            finish_name = gemini_finish_reason_name(response)
            bad_content = (not content) or finish_name == "MAX_TOKENS"
            if bad_content:
                raise EmptyResponseError(
                    f"Gemini returned empty content twice for {paper_id}"
                )

        print(
            f"[{paper_id}] (gemini) Input: {usage['prompt_tokens']} tokens | "
            f"Output: {usage['completion_tokens']} tokens | "
            f"Total: {usage['total_tokens']} tokens"
        )

        return paper_id, paper_data, content, usage


async def generate_paper(prompt_messages: List[Dict], paper_id: str, paper_data: Dict, model_tag: str):
    try:
        if model_tag == "gpt":
            return await generate_gpt(prompt_messages, paper_id, paper_data)
        if model_tag == "haiku":
            return await generate_haiku(prompt_messages, paper_id, paper_data)
        if model_tag == "gemini":
            return await generate_gemini(prompt_messages, paper_id, paper_data)
        raise ValueError(f"Unknown model_tag {model_tag} for paper {paper_id}")
    except Exception as exc:
        print(f"[{paper_id}] generate_paper ({model_tag}) exhausted retries: {exc}")
        return paper_id, paper_data, None, None


def write_prompt_record(file_handle, record: Dict, paper_type: str, hydrated_candidates: List[Dict], prompt_messages: List[Dict], model_tag: str) -> None:
    papers_seen_id: List[str] = []
    for candidate in record["candidate_set"]:
        pair_key = (candidate["author"], candidate["year"])
        resolved_id = SEEN_AUTHOR_YEAR_PAIRS.get(pair_key)
        if resolved_id is not None:
            papers_seen_id.append(resolved_id)

    prompt_record = {
        "id": record["id"],
        "type": paper_type,
        "author": record["author"],
        "year": record["year"],
        "model": model_tag,
        "papers_seen": hydrated_candidates,
        "papers_seen_id": papers_seen_id,
        "prompt": prompt_messages,
    }
    file_handle.write(json.dumps(prompt_record) + "\n")
    file_handle.flush()


def resolve_model_tag_for_record(record: Dict) -> str:
    return record["model_L3"]


def read_presampled_records(node: int) -> List[Dict]:
    records: List[Dict] = []
    input_path = PRESAMPLED_DIR / f"node_{node}_meta.jsonl"
    with input_path.open("r") as meta_file:
        for raw_line in meta_file:
            line = raw_line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


async def generate_node(node: int) -> Dict:
    print("\n" + "=" * 30)
    print(f"Starting node {node}...")
    print("=" * 30 + "\n")

    records = read_presampled_records(node)
    record_by_id = {record["id"]: record for record in records}
    prompt_messages_by_id: Dict[str, List[Dict]] = {}
    paper_data_by_id: Dict[str, Dict] = {}
    model_tag_by_id: Dict[str, str] = {}

    prompts_path = PROMPTS_DIR / f"N_{node}_inputs.jsonl"
    with prompts_path.open("w") as prompts_file:
        for record in records:
            paper_type = record["type"]
            model_tag = resolve_model_tag_for_record(record)
            hydrated_candidates = hydrate_candidate_set(record["candidate_set"])
            user_prompt_content = build_user_prompt(paper_type, hydrated_candidates)
            prompt_messages = build_prompt_messages(user_prompt_content)

            prompt_messages_by_id[record["id"]] = prompt_messages
            model_tag_by_id[record["id"]] = model_tag

            papers_seen_id: List[str] = []
            for candidate in record["candidate_set"]:
                pair_key = (candidate["author"], candidate["year"])
                resolved_id = SEEN_AUTHOR_YEAR_PAIRS.get(pair_key)
                if resolved_id is not None:
                    papers_seen_id.append(resolved_id)

            paper_data_by_id[record["id"]] = {
                "id": record["id"],
                "type": paper_type,
                "author": record["author"],
                "year": record["year"],
                "papers_seen_id": papers_seen_id,
                "model": model_tag,
            }

            write_prompt_record(
                prompts_file,
                record,
                paper_type,
                hydrated_candidates,
                prompt_messages,
                model_tag,
            )

    pending_ids = set(record_by_id.keys())
    results_by_id: Dict[str, Tuple[Dict, str, Dict]] = {}
    attempt = 0

    while pending_ids:
        attempt += 1
        if attempt > MAX_NODE_ATTEMPTS:
            raise RuntimeError(
                f"Node {node} failed to complete after {MAX_NODE_ATTEMPTS} attempts; "
                f"still missing {len(pending_ids)} papers"
            )

        print(
            f"[node {node}] attempt {attempt}: dispatching {len(pending_ids)} paper(s)"
        )
        pending_tasks = []
        for pending_id in pending_ids:
            pending_tasks.append(
                generate_paper(
                    prompt_messages_by_id[pending_id],
                    pending_id,
                    paper_data_by_id[pending_id],
                    model_tag_by_id[pending_id],
                )
            )

        newly_completed: set = set()
        for task in asyncio.as_completed(pending_tasks):
            try:
                paper_id, paper_data, json_response, usage = await task
            except Exception as exc:
                print(f"[node {node}] attempt {attempt} task error: {exc}")
                continue
            if not json_response:
                print(
                    f"[node {node}] attempt {attempt} empty response for "
                    f"{paper_id}, will re-queue"
                )
                continue
            results_by_id[paper_id] = (paper_data, json_response, usage)
            newly_completed.add(paper_id)

        pending_ids -= newly_completed

        if pending_ids:
            backoff = min(60, 5 * attempt)
            print(
                f"[node {node}] {len(pending_ids)} papers still missing after "
                f"attempt {attempt}; sleeping {backoff}s before re-queue"
            )
            await asyncio.sleep(backoff)

    node_citations = defaultdict(int)
    hallucinations: List[Dict] = []
    over_cap_violations: List[Dict] = []
    token_records: List[Dict] = []
    node_output_dir = OUTPUT_DIR / f"node_{node}"
    node_output_dir.mkdir(parents=True, exist_ok=True)

    with (node_output_dir / f"node_{node}.jsonl").open("w") as papers_file:
        for paper_id, (paper_data, json_response, usage) in results_by_id.items():
            parsed = json.loads(json_response)
            title = parsed.get("title", "").strip()
            abstract = parsed.get("abstract", "").strip()
            body = parsed.get("body", "").strip()

            raw_citations = get_citations(body)
            citation_id_set: set = set()
            for cite in raw_citations:
                if cite in SEEN_AUTHOR_YEAR_PAIRS:
                    citation_id_set.add(SEEN_AUTHOR_YEAR_PAIRS[cite])
                else:
                    print(
                        f"Warning: Potential hallucinated citation {paper_id} - {cite}"
                    )
                    hallucinations.append({paper_id: list(cite)})

            if len(citation_id_set) > CITATION_CAP:
                print(
                    f"[OVER CAP] {paper_id}: {len(citation_id_set)} unique citations, "
                    f"truncating to {CITATION_CAP}"
                )
                over_cap_violations.append(
                    {
                        "paper_id": paper_id,
                        "node": node,
                        "original_count": len(citation_id_set),
                        "original_citation_ids": list(citation_id_set),
                    }
                )
                citation_id_set = set(list(citation_id_set)[:CITATION_CAP])

            for cited_id in citation_id_set:
                node_citations[cited_id] += 1

            if usage:
                prompt_tokens = usage["prompt_tokens"]
                completion_tokens = usage["completion_tokens"]
                total_tokens = usage["total_tokens"]
            else:
                prompt_tokens = 0
                completion_tokens = 0
                total_tokens = 0

            paper_record = {
                "id": paper_id,
                "model": paper_data.get("model"),
                "author": paper_data.get("author"),
                "year": paper_data.get("year"),
                "type": paper_data.get("type"),
                "title": title,
                "abstract": abstract,
                "body": body,
                "papers_seen_id": paper_data.get("papers_seen_id"),
                "citations": [list(cite) for cite in raw_citations],
                "citation_ids": list(citation_id_set),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }
            papers_file.write(json.dumps(paper_record) + "\n")
            papers_file.flush()

            token_records.append(
                {
                    "id": paper_id,
                    "model": paper_data.get("model"),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                }
            )

            GENERATED_BY_ID[paper_id] = {
                "author": paper_data.get("author"),
                "year": paper_data.get("year"),
                "title": title,
                "abstract": abstract,
            }

    print("\n" + "=" * 30)
    print(f"Node {node} Statistics")
    print("=" * 30 + "\n")

    with (node_output_dir / f"node_{node}_stats.jsonl").open("w") as stats_file:
        for cited_id, count in node_citations.items():
            stats_file.write(json.dumps({cited_id: count}) + "\n")
            print(f"{cited_id}: {count}")

    with (CITATION_COUNTS_DIR / "hallucinations.jsonl").open("a") as hall_file:
        for item in hallucinations:
            hall_file.write(json.dumps(item) + "\n")

    with (CITATION_COUNTS_DIR / "over_cap_violations.jsonl").open("a") as overcap_file:
        for item in over_cap_violations:
            overcap_file.write(json.dumps(item) + "\n")

    with (node_output_dir / f"node_{node}_token_usage.jsonl").open("w") as tu_file:
        for record in token_records:
            tu_file.write(json.dumps(record) + "\n")

    node_prompt_total = 0
    node_completion_total = 0
    node_total = 0
    for record in token_records:
        node_prompt_total += record["prompt_tokens"]
        node_completion_total += record["completion_tokens"]
        node_total += record["total_tokens"]

    node_token_totals = {
        "node": node,
        "total_prompt_tokens": node_prompt_total,
        "total_completion_tokens": node_completion_total,
        "total_tokens": node_total,
    }

    with (node_output_dir / f"node_{node}_token_totals.jsonl").open("w") as tt_file:
        tt_file.write(json.dumps(node_token_totals) + "\n")

    print(
        f"\nNode {node} Token Usage: "
        f"Input: {node_prompt_total} | "
        f"Output: {node_completion_total} | "
        f"Total: {node_total}"
    )

    for cited_id, count in node_citations.items():
        CITATION_COUNTS[cited_id] += count

    return node_token_totals


def bootstrap_generated_by_id_from_seed(seed_records: List[Dict]) -> None:
    for seed_paper in seed_records:
        GENERATED_BY_ID[seed_paper["id"]] = {
            "author": seed_paper["author"],
            "year": seed_paper["year"],
            "title": seed_paper["title"],
            "abstract": seed_paper["abstract"],
        }


async def run_experiment() -> None:
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SEED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MASTER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CITATION_COUNTS_DIR.mkdir(parents=True, exist_ok=True)
    (CITATION_COUNTS_DIR / "hallucinations.jsonl").open("w").close()
    (CITATION_COUNTS_DIR / "over_cap_violations.jsonl").open("w").close()

    seed_records = load_seed_records()
    bootstrap_generated_by_id_from_seed(seed_records)

    kv_pairs = load_kv_pairs()
    SEEN_AUTHOR_YEAR_PAIRS.update(kv_pairs)

    all_node_token_totals: List[Dict] = []
    for node in range(TOTAL_NODES):
        node_token_totals = await generate_node(node)
        all_node_token_totals.append(node_token_totals)

    with (CITATION_COUNTS_DIR / "citation_counts.jsonl").open("w") as f:
        for cited_id, count in CITATION_COUNTS.items():
            f.write(json.dumps({cited_id: count}) + "\n")

    with (MASTER_OUTPUT_DIR / "kv_pairs.jsonl").open("w") as f:
        for (author, year), paper_id in SEEN_AUTHOR_YEAR_PAIRS.items():
            f.write(
                json.dumps({"author": author, "year": year, "id": paper_id}) + "\n"
            )

    with (MASTER_OUTPUT_DIR / "token_usage.jsonl").open("w") as f:
        for totals in all_node_token_totals:
            f.write(json.dumps(totals) + "\n")

    print("Finished generating nodes")


if __name__ == "__main__":
    asyncio.run(run_experiment())
