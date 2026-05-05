import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
#from sentence_transformers import SentenceTransformer
#from sentence_transformers.util import cos_sim

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent
SEED_PATH = REPO_ROOT / "gpt" / "output" / "seed" / "seed.jsonl"
NODE_ROOT = REPO_ROOT / "gpt" / "output"
OUTPUT_DIR = SCRIPT_PATH.parent
TOPIC_TEXT = "Knowledge distillation or model compression in deep learning or NLP"
#EMBEDDING_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
NODE_RANGE = range(0, 12)


def load_jsonl(path):
    jsonl_path = Path(path)
    records = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            records.append(json.loads(stripped))
    return records


def compute_seed_relevance(seed_records, topic_text):

    seed_ids = []
    abstracts = []
    for record in seed_records:
        seed_ids.append(record["id"])
        abstracts.append(record["abstract"])

    vectorizer = TfidfVectorizer()
    vectorizer.fit(abstracts + [topic_text])

    topic_vec = vectorizer.transform([topic_text])
    abstract_vecs = vectorizer.transform(abstracts)

    relevance_scores = cosine_similarity(topic_vec, abstract_vecs).flatten()

    relevance_by_seed = {}
    for index in range(len(seed_ids)):
        relevance_by_seed[seed_ids[index]] = float(relevance_scores[index])
        
    return relevance_by_seed


def extract_seeds_from_prompt(papers_seen_id, seed_relevance):
    seeds_in_prompt = []
    for paper_id in papers_seen_id:
        if not paper_id.startswith("SEED_"):
            continue
        if paper_id not in seed_relevance:
            continue
        seeds_in_prompt.append(paper_id)
    return seeds_in_prompt


def build_regression_rows(node_records, node_index, seed_relevance):
    rows = []
    for record in node_records:
        prompt_id = record["id"]
        prompt_type = record["type"]
        papers_seen = record.get("papers_seen_id", [])
        citation_ids = record.get("citation_ids", [])
        citation_set = set(citation_ids)

        seeds_in_prompt = extract_seeds_from_prompt(papers_seen, seed_relevance)
        n_seeds_in_prompt = len(seeds_in_prompt)

        for seed_id in seeds_in_prompt:
            cited_flag = 1 if seed_id in citation_set else 0
            row = {
                "prompt_id": prompt_id,
                "seed_id": seed_id,
                "cited": cited_flag,
                "n_other_seeds": n_seeds_in_prompt - 1,
                "prompt_type": prompt_type,
                "relevance": seed_relevance[seed_id],
                "node": node_index,
            }
            rows.append(row)
    return rows


def build_full_dataframe(node_root, seed_relevance):
    all_rows = []
    citation_length_sum = 0
    citation_length_count = 0

    node_root_path = Path(node_root)
    for node_index in NODE_RANGE:
        node_dir = node_root_path / "node_{}".format(node_index)
        node_path = node_dir / "node_{}.jsonl".format(node_index)
        node_records = load_jsonl(node_path)

        for record in node_records:
            citation_length_sum += len(record.get("citation_ids", []))
            citation_length_count += 1

        node_rows = build_regression_rows(node_records, node_index, seed_relevance)
        all_rows.extend(node_rows)

    dataframe = pd.DataFrame(all_rows)

    if citation_length_count > 0:
        mean_citation_length = citation_length_sum / citation_length_count
    else:
        mean_citation_length = 0.0

    return dataframe, mean_citation_length


def fit_logit_clustered(dataframe):
    formula = "cited ~ n_other_seeds + relevance + C(prompt_type) + C(node)"
    model = smf.logit(formula=formula, data=dataframe)
    cluster_groups = dataframe["prompt_id"].values
    result = model.fit(
        disp=False,
        cov_type="cluster",
        cov_kwds={"groups": cluster_groups},
    )
    print(result.summary())
    return result


def save_odds_ratios(result, out_path):
    terms = list(result.params.index)
    coefficients = result.params.values
    p_values = result.pvalues.values
    odds_ratios = np.exp(coefficients)

    odds_frame = pd.DataFrame({
        "term": terms,
        "coef": coefficients,
        "odds_ratio": odds_ratios,
        "p_value": p_values,
    })
    odds_frame.to_csv(out_path, index=False)
    return odds_frame


def plot_marginal_effect(result, dataframe, out_path):
    mean_relevance = float(dataframe["relevance"].mean())

    observed_min = int(dataframe["n_other_seeds"].min())
    observed_max = int(dataframe["n_other_seeds"].max())
    competitor_counts = list(range(observed_min, observed_max + 1))
    node_values = list(NODE_RANGE)

    prediction_rows = []
    for competitor_count in competitor_counts:
        for node_value in node_values:
            prediction_rows.append({
                "n_other_seeds": competitor_count,
                "relevance": mean_relevance,
                "prompt_type": "literature review",
                "node": node_value,
            })
    prediction_frame = pd.DataFrame(prediction_rows)

    prediction = result.get_prediction(prediction_frame)
    summary_frame = prediction.summary_frame(alpha=0.05)

    mean_candidates = ["mean", "predicted"]
    lower_candidates = ["mean_ci_lower", "ci_lower", "predicted_ci_lower"]
    upper_candidates = ["mean_ci_upper", "ci_upper", "predicted_ci_upper"]

    mean_column = None
    for candidate in mean_candidates:
        if candidate in summary_frame.columns:
            mean_column = candidate
            break
    lower_column = None
    for candidate in lower_candidates:
        if candidate in summary_frame.columns:
            lower_column = candidate
            break
    upper_column = None
    for candidate in upper_candidates:
        if candidate in summary_frame.columns:
            upper_column = candidate
            break

    if mean_column is None or lower_column is None or upper_column is None:
        raise RuntimeError(
            "Unexpected prediction summary columns: {}".format(list(summary_frame.columns))
        )

    summary_frame = summary_frame.copy()
    summary_frame["n_other_seeds"] = prediction_frame["n_other_seeds"].values

    averaged_mean = []
    averaged_lower = []
    averaged_upper = []
    for competitor_count in competitor_counts:
        subset = summary_frame[summary_frame["n_other_seeds"] == competitor_count]
        averaged_mean.append(float(subset[mean_column].mean()))
        averaged_lower.append(float(subset[lower_column].mean()))
        averaged_upper.append(float(subset[upper_column].mean()))

    predicted_mean = np.array(averaged_mean)
    ci_lower = np.array(averaged_lower)
    ci_upper = np.array(averaged_upper)

    figure, axis = plt.subplots(figsize=(5.5, 3.5))
    axis.plot(
        competitor_counts,
        predicted_mean,
        marker="o",
        color="#2b6cb0",
    )
    axis.fill_between(
        competitor_counts,
        ci_lower,
        ci_upper,
        color="#2b6cb0",
        alpha=0.2,
    )
    axis.set_xlabel("Seed competitor count")
    axis.set_ylabel("P(cited | shown)")
    axis.set_ylim(0, 1)
    x_max = competitor_counts[-1]
    major_xticks = list(range(0, x_max + 1, 5))
    if major_xticks[-1] != x_max:
        major_xticks.append(x_max)
    axis.set_xticks(major_xticks)
    axis.set_xticks(competitor_counts, minor=True)

    marginal_df = pd.DataFrame({
        "n_other_seeds": competitor_counts,
        "predicted_mean": predicted_mean,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    })
    marginal_df.to_csv(out_path.with_suffix(".csv"), index=False)

    figure.tight_layout()
    figure.savefig(out_path, dpi=300)
    plt.close(figure)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading seed metadata from {}".format(SEED_PATH))
    seed_records = load_jsonl(SEED_PATH)
    print("Loaded {} seed records".format(len(seed_records)))

    print("Computing seed relevance scores vs topic: {!r}".format(TOPIC_TEXT))
    seed_relevance = compute_seed_relevance(seed_records, TOPIC_TEXT)

    print("Building regression dataframe across nodes {}-{}".format(NODE_RANGE.start, NODE_RANGE.stop - 1))
    dataframe, mean_citation_length = build_full_dataframe(NODE_ROOT, seed_relevance)
    print("Regression dataframe rows: {}".format(len(dataframe)))
    print("Mean citation_ids length: {:.3f}".format(mean_citation_length))

    regression_csv_path = OUTPUT_DIR / "regression_data.csv"
    dataframe.to_csv(regression_csv_path, index=False)
    print("Saved regression dataframe to {}".format(regression_csv_path))

    print("\nFitting logistic regression with clustered standard errors on prompt_id")
    result = fit_logit_clustered(dataframe)

    odds_csv_path = OUTPUT_DIR / "odds_ratios.csv"
    save_odds_ratios(result, odds_csv_path)
    print("\nSaved odds ratios to {}".format(odds_csv_path))

    plot_path = OUTPUT_DIR / "marginal_effect.png"
    plot_marginal_effect(result, dataframe, plot_path)
    print("Saved marginal effect plot to {}".format(plot_path))


if __name__ == "__main__":
    main()
