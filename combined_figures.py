import json
import numpy as np
import matplotlib.pyplot as plt


GPT_COLOR = "#10A37F"
CLAUDE_COLOR = "#D97757"
RANDOM_COLOR = "#888888"

MAX_NODES = 12

GPT_CONC = "gpt/output/master/concentration.jsonl"
GPT_CONC_RND = "gpt/output/master/concentration_random_baseline.jsonl"
CLAUDE_CONC = "claude/output/master/concentration.jsonl"
CLAUDE_CONC_RND = "claude/output/master/concentration_random_baseline.jsonl"

GPT_RATE = "gpt/output/master/seed_vs_llm_citation_rate.jsonl"
CLAUDE_RATE = "claude/output/master/seed_vs_llm_citation_rate.jsonl"


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def plot_concentration():
    gpt_llm = {r["node"]: r["top_10pct_share"] for r in load_jsonl(GPT_CONC)}
    gpt_rnd = {r["node"]: r["top_10pct_share_mean"] for r in load_jsonl(GPT_CONC_RND)}
    claude_llm = {r["node"]: r["top_10pct_share"] for r in load_jsonl(CLAUDE_CONC)}
    claude_rnd = {r["node"]: r["top_10pct_share_mean"] for r in load_jsonl(CLAUDE_CONC_RND)}

    nodes = list(range(MAX_NODES))
    x = np.arange(len(nodes))

    all_vals = (
        [gpt_llm[n] for n in nodes] + [gpt_rnd[n] for n in nodes]
        + [claude_llm[n] for n in nodes] + [claude_rnd[n] for n in nodes]
    )
    y_top = min(100, max(all_vals) * 1.15)

    fig, (ax_gpt, ax_claude) = plt.subplots(
        1, 2, figsize=(14, 5), constrained_layout=True
    )

    for ax, llm_data, rnd_data, color, model_name, label in [
        (ax_gpt, gpt_llm, gpt_rnd, GPT_COLOR, "GPT 5 Mini", "(a)"),
        (ax_claude, claude_llm, claude_rnd, CLAUDE_COLOR, "Claude Haiku 4.5", "(b)"),
    ]:
        llm_y = [llm_data[n] for n in nodes]
        rnd_y = [rnd_data[n] for n in nodes]

        ax.plot(x, llm_y, color=color, marker="o", linewidth=2, linestyle="-",
                label=model_name, markersize=5)
        ax.plot(x, rnd_y, color=RANDOM_COLOR, marker="s", linewidth=2,
                linestyle="--", label="Random baseline", markersize=5)

        ax.set_xlabel("Node", fontsize=10)
        ax.set_ylabel("Top 10% citation share (%)", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(nodes, fontsize=9)
        ax.tick_params(axis="y", labelsize=9)
        local_y = llm_y + rnd_y
        ax.set_ylim(0, min(100, max(local_y) * 1.15))
        ax.set_title(label, fontsize=11)
        ax.legend(loc="upper left", fontsize=9, frameon=True)

    fig.savefig("./figures/combined_concentration.png", dpi=300, bbox_inches="tight")
    print("Saved combined_concentration.png")
    plt.close(fig)


def plot_citation_rate():
    gpt_rows = {r["node"]: r for r in load_jsonl(GPT_RATE)}
    claude_rows = {r["node"]: r for r in load_jsonl(CLAUDE_RATE)}

    nodes = list(range(MAX_NODES))
    x = np.arange(len(nodes))

    all_vals = []
    for data in (gpt_rows, claude_rows):
        for n in nodes:
            if data[n]["seed_rate"] is not None:
                all_vals.append(100 * data[n]["seed_rate"])
            if data[n]["llm_rate"] is not None:
                all_vals.append(100 * data[n]["llm_rate"])
    y_top = max(all_vals) * 1.15 if all_vals else 10

    fig, (ax_gpt, ax_claude) = plt.subplots(
        1, 2, figsize=(14, 5), constrained_layout=True
    )

    for ax, data, color, model_name, label in [
        (ax_gpt, gpt_rows, GPT_COLOR, "GPT 5 Mini", "(a)"),
        (ax_claude, claude_rows, CLAUDE_COLOR, "Claude Haiku 4.5", "(b)"),
    ]:
        seed_y = [100 * data[n]["seed_rate"] if data[n]["seed_rate"] is not None else 0 for n in nodes]
        llm_y = [100 * data[n]["llm_rate"] if data[n]["llm_rate"] is not None else 0 for n in nodes]

        ax.plot(x, seed_y, color=color, marker="o", linewidth=2, linestyle="-",
                label="Seed papers", markersize=5)
        ax.plot(x, llm_y, color=color, marker="^", linewidth=2, linestyle="--",
                label="LLM-generated papers", markersize=5)

        ax.set_xlabel("Node", fontsize=10)
        ax.set_ylabel("Citation rate (%)", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(nodes, fontsize=9)
        ax.tick_params(axis="y", labelsize=9)
        local_y = seed_y + llm_y
        ax.set_ylim(0, max(local_y) * 1.15 if local_y else 10)
        ax.set_title(label, fontsize=11)
        ax.legend(loc="best", fontsize=9, frameon=True)

    fig.savefig("./figures/combined_citation_rate.png", dpi=300, bbox_inches="tight")
    print("Saved combined_citation_rate.png")
    plt.close(fig)


if __name__ == "__main__":
    plot_concentration()
    plot_citation_rate()
