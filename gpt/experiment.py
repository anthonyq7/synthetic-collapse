import asyncio
import json
import os
import random
import re
from dotenv import load_dotenv
from openai import AsyncOpenAI
from typing import List, Dict
from collections import defaultdict
from faker import Faker
from citation_parser import extract_citations_from_body

load_dotenv()

NODE_SIZE = 120
STRATUM_SIZE = 60
CITATION_CAP = 10
POOLED_PAPERS = []
TOTAL_NODES = 12
MODEL = "gpt-5-mini"
MAX_CONCURRENT = 120
TARGET_LENGTH = 500
SEED = 42
PAPER_SET_LENGTH = 30
TOPIC = "Knowledge distillation or model compression in deep learning or NLP"
client = AsyncOpenAI()
CITATION_COUNTS = defaultdict(int)
POSSIBLE_PAPER_IDS = set()
SEEN_AUTHOR_YEAR_PAIRS = {}

fake = Faker()
Faker.seed(SEED)
_SIMPLE_NAME = re.compile(r"^[A-Z][a-z]+$")


def _get_author(paper: dict) -> str:
    return paper["author"]

def simple_surname() -> str:
    name = fake.last_name()
    while not _SIMPLE_NAME.match(name):
        name = fake.last_name()
    return name

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

def create_set():
    return random.sample(POOLED_PAPERS, PAPER_SET_LENGTH)

def generate_prompts(node: int):

    with open(f"gpt/prompts/N_{node}_inputs.jsonl", "w") as f:

        pos_indicies = set(random.sample(range(NODE_SIZE-1), STRATUM_SIZE))

        for i in range(NODE_SIZE):
            paper_id = f"N{node}P{i}"
            paper_type = "literature review"
            random_papers = create_set()
            papers_seen_id = set()
            author = simple_surname()
            year = random.randint(2017, 2025)
            while (author, year) in SEEN_AUTHOR_YEAR_PAIRS:
                author = simple_surname()
                year = random.randint(2017, 2025)

            SEEN_AUTHOR_YEAR_PAIRS[(author, year)] = paper_id

            for p in random_papers:
                key = (p["author"], p["year"])
                if key in SEEN_AUTHOR_YEAR_PAIRS:
                    papers_seen_id.add(SEEN_AUTHOR_YEAR_PAIRS[key])

            if i in pos_indicies:
                paper_type = "position paper"
                user_prompt_content = f"""
                    Using only the articles provided, argue a position on the following topic.
                    Support claims by citing relevant articles inline.
                    Only cite articles from the provided list using their exact author and year via inline citations.
                    You MUST cite no more than {CITATION_CAP} unique articles in the body.

                    Topic: {TOPIC}

                    Articles:
                    {random_papers}
                    """
            else:
                paper_type = "literature review"
                user_prompt_content = f"""
                    Using only the articles provided, synthesize what is known about the following topic.
                    Support claims by citing relevant articles inline.
                    Only cite articles from the provided list using their exact author and year via inline citations.
                    You MUST cite no more than {CITATION_CAP} unique articles in the body.

                    Topic: {TOPIC}

                    Articles:
                    {random_papers}
                    """

            prompt_messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": user_prompt_content
                }
            ]

            paper_record = {
                "id": paper_id,
                "type": paper_type,
                "author": author,
                "year": year,
                "papers_seen": random_papers,
                "papers_seen_id": list(papers_seen_id),
                "prompt": prompt_messages
            }
            f.write(json.dumps(paper_record) + "\n")
            f.flush()

async def generate_paper(prompt: str, paper_id: str, paper_data: dict, semaphore = asyncio.Semaphore(MAX_CONCURRENT)):
    
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model = MODEL,
                messages = prompt,
                max_completion_tokens=5000,
                response_format={"type": "json_object"}
            )

            usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
            if response.usage:
                usage["prompt_tokens"] += response.usage.prompt_tokens
                usage["completion_tokens"] += response.usage.completion_tokens
                usage["total_tokens"] += response.usage.total_tokens

            content = response.choices[0].message.content
            if not content:
                print(f"Empty response {paper_id}")
                print(f"Retrying....")
                await asyncio.sleep(5)

                response = await client.chat.completions.create(
                    model = MODEL,
                    messages = prompt,
                    max_completion_tokens=10000,
                    response_format={"type": "json_object"}
                )

                if response.usage:
                    usage["prompt_tokens"] += response.usage.prompt_tokens
                    usage["completion_tokens"] += response.usage.completion_tokens
                    usage["total_tokens"] += response.usage.total_tokens

                content = response.choices[0].message.content
                if not content:
                    return paper_id, paper_data, None, usage
            
            print(
                f"[{paper_id}] Input: {usage['prompt_tokens']} tokens | "
                f"Output: {usage['completion_tokens']} tokens | "
                f"Total: {usage['total_tokens']} tokens"
            )
            
            return paper_id, paper_data, content, usage

        except Exception as e:
            print(f"API Error: {e}")
            print(f"Paper: {paper_id}")
            await asyncio.sleep(5)
            return paper_id, paper_data, None, None

async def generate_node(node: int):

    print("\n")
    print(30*"=")
    print(f"Starting node {node}...")
    print(30*"=")
    print("\n")

    generate_prompts(node)

    wrapped_tasks = []
    node_citations = defaultdict(int)
    hallucinations = []
    over_cap_violations = []
    token_records = []

    with open(f"gpt/prompts/N_{node}_inputs.jsonl", "r") as f:
        for raw_line in f:
            line = json.loads(raw_line)
            paper_data = {
                "id": line.get("id"),
                "type": line.get("type"),
                "author": line.get("author"),
                "year": line.get("year"),
                "papers_seen_id": line.get("papers_seen_id")
            }

            prompt = line.get("prompt")
            paper_id = str(line.get("id"))
            wrapped_tasks.append(
                generate_paper(prompt=prompt, paper_id=paper_id, paper_data=paper_data)
            )

    with open(f"gpt/output/node_{node}/node_{node}.jsonl", "w") as f:
        for task in asyncio.as_completed(wrapped_tasks):
            paper_id, paper_data, json_response, usage = await task

            if not json_response:
                print(f"{paper_id} was empty...")
                continue

            print(f"Finished {paper_id}...")

            json_response = json.loads(json_response)

            title = json_response.get("title", "").strip()
            abstract = json_response.get("abstract", "").strip()
            body = json_response.get("body", "").strip()
            
            raw_citations = get_citations(body)
            citation_id_set = set()

            for cite in raw_citations:
                if cite in SEEN_AUTHOR_YEAR_PAIRS:
                    citation_id_set.add(SEEN_AUTHOR_YEAR_PAIRS[cite])
                else:
                    print(
                        f"Warning: Potential hallucinated citation {paper_id} - {cite}"
                    )
                    hallucinations.append({paper_id: cite})
                    

            if len(citation_id_set) > CITATION_CAP:
                print(
                    f"[OVER CAP] {paper_id}: {len(citation_id_set)} unique citations, "
                    f"truncating to {CITATION_CAP}"
                )
                over_cap_violations.append({
                    "paper_id": paper_id,
                    "node": node,
                    "original_count": len(citation_id_set),
                    "original_citation_ids": list(citation_id_set),
                })
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
                "author": paper_data.get("author"),
                "year": paper_data.get("year"),
                "type": paper_data.get("type"),
                "title": title,
                "abstract": abstract,
                "body": body,
                "papers_seen_id": paper_data.get("papers_seen_id"),
                "citations": list(raw_citations),
                "citation_ids": list(citation_id_set),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }

            token_records.append({
                "id": paper_id,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            })

            new_paper = {
                "author": paper_data.get("author"),
                "year": paper_data.get("year"),
                "title": title,
                "abstract": abstract
            }

            POOLED_PAPERS.append(new_paper)

            f.write(json.dumps(paper_record) + "\n")
            f.flush()

    print("\n")
    print(30*"=")
    print(f"Node {node} Statistics")
    print(30*"=")
    print("\n")

    with open(f"gpt/output/node_{node}/node_{node}_stats.jsonl", "w") as f:
        for k, v in node_citations.items():
            f.write(json.dumps({k:v}) + "\n")
            print(f"{k}: {v}")
        
        f.flush()
    
    with open(f"gpt/output/citation_counts/hallucinations.jsonl", "a") as f:
        for item in hallucinations:
            f.write(json.dumps(item) + "\n")
        f.flush()

    with open(f"gpt/output/citation_counts/over_cap_violations.jsonl", "a") as f:
        for item in over_cap_violations:
            f.write(json.dumps(item) + "\n")
        f.flush()

    with open(f"gpt/output/node_{node}/node_{node}_token_usage.jsonl", "w") as f:
        for record in token_records:
            f.write(json.dumps(record) + "\n")
        f.flush()

    node_prompt_total = sum(
        r["prompt_tokens"] for r in token_records
    )
    node_completion_total = sum(
        r["completion_tokens"] for r in token_records
    )
    node_total = sum(
        r["total_tokens"] for r in token_records
    )

    node_token_totals = {
        "node": node,
        "total_prompt_tokens": node_prompt_total,
        "total_completion_tokens": node_completion_total,
        "total_tokens": node_total
    }

    with open(f"gpt/output/node_{node}/node_{node}_token_totals.jsonl", "w") as f:
        f.write(json.dumps(node_token_totals) + "\n")
        f.flush()

    print(
        f"\nNode {node} Token Usage: "
        f"Input: {node_prompt_total} | "
        f"Output: {node_completion_total} | "
        f"Total: {node_total}"
    )

    for k, v in node_citations.items():
        CITATION_COUNTS[k] += v

    POOLED_PAPERS.sort(key=_get_author)

    return node_token_totals

def get_seed():
    try:
        pooled = []
        with open("buckets/bucket_50_500.jsonl") as f:
            for paper in f:
                pooled.append(json.loads(paper))
            
        return pooled
    except FileNotFoundError as e:
        print(f"buckets/bucket_50_500.jsonl not found...")
        return None

def standardize_seed() -> List[Dict]:
    seed = get_seed()
    if not seed:
        return []

    papers = seed

    arxiv_list = []
    arxiv_citation_count = []
    return_list = []

    for i, paper in enumerate(papers):
        paper_id = f"SEED_{i}"
        title = paper.get("title")
        abstract = paper.get("abstract")
        citation_count = paper.get("citation_count")
        fake_surname = simple_surname()
        fake_year = random.randint(2017, 2022)

        while (fake_surname, fake_year) in SEEN_AUTHOR_YEAR_PAIRS:
                fake_surname = simple_surname()
                fake_year = random.randint(2017, 2022)

        SEEN_AUTHOR_YEAR_PAIRS[(fake_surname, fake_year)] = paper_id

        output_object = {
            "id": paper_id,
            "author": fake_surname,
            "year": fake_year,
            "title": title,
            "abstract": abstract
        }

        return_object = {
            "author": fake_surname,
            "year": fake_year,
            "title": title,
            "abstract": abstract
        }

        citation_object = {
            "id": paper_id,
            "citation_count": citation_count
        }

        arxiv_list.append(output_object)
        return_list.append(return_object)
        arxiv_citation_count.append(citation_object)
        POSSIBLE_PAPER_IDS.add(paper_id)
        SEEN_AUTHOR_YEAR_PAIRS[(fake_surname, fake_year)] = paper_id
    
    with open("gpt/output/seed/seed_initial.jsonl", "w") as f:
        for item in arxiv_citation_count:
            f.write(json.dumps(item) + "\n")

    with open("gpt/output/seed/seed.jsonl", "w") as f:
        for item in arxiv_list:
            f.write(json.dumps(item) + "\n")   

    return return_list

def get_citations(text: str):
    return extract_citations_from_body(text)


async def run_experiment():

    random.seed(SEED)

    #Make output directories 
    os.makedirs("gpt/output/seed", exist_ok=True)
    os.makedirs("gpt/output/master", exist_ok=True)
    os.makedirs("gpt/output/citation_counts", exist_ok=True)
    os.makedirs("gpt/prompts", exist_ok=True)
    open("gpt/output/citation_counts/hallucinations.jsonl", "w").close()
    open("gpt/output/citation_counts/over_cap_violations.jsonl", "w").close()

    #standardizes arXiv papers and adds them to the pool
    #Additionally, saves the initial citation counts + add another running citation count to citation_counts
    POOLED_PAPERS.extend(standardize_seed())

    for i in range(TOTAL_NODES):
        os.makedirs(f"gpt/output/node_{i}", exist_ok=True)
    
    all_node_token_totals = []

    for i in range(TOTAL_NODES):
        node_token_totals = await generate_node(i)
        all_node_token_totals.append(node_token_totals)

    with open("gpt/output/citation_counts/citation_counts.jsonl", "w") as f:
        for k, v in CITATION_COUNTS.items():
            f.write(json.dumps({k: v}) + "\n")
    
    with open("gpt/output/master/kv_pairs.jsonl", "w") as f:
        for (author, year), paper_id in SEEN_AUTHOR_YEAR_PAIRS.items():
            record = {"author": author, "year": year, "id": paper_id}
            f.write(json.dumps(record) + "\n")

    with open("gpt/output/master/token_usage.jsonl", "w") as f:
        for totals in all_node_token_totals:
            f.write(json.dumps(totals) + "\n")

    print("Finished generating nodes")

if __name__ == "__main__":
    asyncio.run(run_experiment())
