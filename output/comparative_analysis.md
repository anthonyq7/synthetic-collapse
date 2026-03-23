# Comparative Analysis: LLM vs Random Citation Experiment
## alpha_10_cap — 12 Nodes · 120 Papers/Node · 120 Seeds

---

## Experimental Setup

Two citation experiments run over the same corpus, same structure:

| Parameter | Value |
|-----------|-------|
| Seed papers | 120 (arXiv, 50–500 citations, pre-Nov 2022) |
| Nodes | 12 |
| Papers per node | 120 (60 lit review / 60 position paper) |
| Context window | 30 papers shown per generated paper |
| Total generated papers | 1,440 |
| Total papers in pool by end | 1,560 |

**LLM experiment**: GPT model reads the 30-paper context and writes a paper, citing from that set at its own discretion.

**Random experiment**: A controlled baseline using the *same* `papers_seen_id` lists (same 30 papers were shown to each paper) but citations are assigned uniformly at random, drawing the same total citation count per node as the LLM produced. This isolates structural effects (first-mover advantage, pool growth) from LLM-specific behavior.

---

## Metric 1: Concentration (Gini Coefficient + Top-N Share)

### Data

| Node | Available | LLM Gini | RND Gini | Δ Gini | LLM Top-5% | RND Top-5% | Uniform Top-5% |
|------|-----------|-----------|-----------|--------|------------|------------|----------------|
| 0    | 120       | 0.1416    | 0.0995    | +0.042 | 6.19%      | 5.87%      | 4.17%          |
| 1    | 240       | 0.1871    | 0.1477    | +0.039 | 3.65%      | 3.35%      | 2.08%          |
| 2    | 360       | 0.2349    | 0.1920    | +0.043 | 2.79%      | 2.79%      | 1.39%          |
| 3    | 480       | 0.2733    | 0.2315    | +0.042 | 2.59%      | 2.41%      | 1.04%          |
| 4    | 600       | 0.3132    | 0.2620    | +0.051 | 2.24%      | 2.09%      | 0.83%          |
| 5    | 720       | 0.3094    | 0.2863    | +0.023 | 1.94%      | 1.94%      | 0.69%          |
| 6    | 840       | 0.3300    | 0.3065    | +0.024 | 1.68%      | 1.61%      | 0.60%          |
| 7    | 960       | 0.3399    | 0.3119    | +0.028 | 1.58%      | 1.62%      | 0.52%          |
| 8    | 1,080     | 0.3818    | 0.3446    | +0.037 | 1.59%      | 1.44%      | 0.46%          |
| 9    | 1,200     | 0.4063    | 0.3710    | +0.035 | 1.70%      | 1.51%      | 0.42%          |
| 10   | 1,320     | 0.3975    | 0.3788    | +0.019 | 1.40%      | 1.44%      | 0.38%          |
| 11   | 1,440     | 0.4373    | 0.3993    | +0.038 | 1.37%      | 1.25%      | 0.35%          |

### Finding: Most Gini Growth Is Structural, but LLM Adds a Consistent Premium

The random baseline Gini rises from 0.10 to 0.40 — nearly as much as the LLM's 0.14 to 0.44. This is a critical finding: **the majority of the rising Gini is not caused by LLM selectivity, but by the structural first-mover advantage of seed papers being citeable across all 12 generations**. A paper available for 12 nodes of 120 citations each will naturally accumulate far more than a paper added at node 10.

The LLM's additional contribution is a consistent **+0.02 to +0.05 Gini premium** above the random baseline, present at every node. This gap never closes. The LLM is genuinely more concentrating than random citation, but the effect size is moderate — it adds roughly 10% more inequality on top of what structural dynamics already produce.

**Top-5% shares** tell the same story: both LLM and random remain persistently above the uniform baseline at every node, confirming preferential concentration exists in both. The LLM's excess over random is small (0–0.3 percentage points) but directionally consistent.

---

## Metric 2: False Negatives (Shown-but-Not-Cited Rate)

A false negative occurs when a paper is included in a generated paper's 30-paper context window but is never cited by that paper. Two measures:
- **Paper-level FN rate**: fraction of shown papers that received zero citations
- **Event-level FN rate**: fraction of all individual exposure events that produced no citation

### Data

| Node | LLM Paper FN | RND Paper FN | LLM Event FN | RND Event FN | Event Excess |
|------|-------------|-------------|-------------|-------------|--------------|
| 0    | 0.000       | 0.000       | 0.000       | 0.000       | +0.000       |
| 1    | 0.004       | 0.000       | 0.002       | 0.000       | +0.002       |
| 2    | 0.000       | 0.000       | 0.000       | 0.000       | +0.000       |
| 3    | 0.015       | 0.006       | 0.014       | 0.001       | +0.013       |
| 4    | 0.027       | 0.007       | 0.019       | 0.003       | +0.016       |
| 5    | 0.041       | 0.010       | 0.027       | 0.003       | +0.024       |
| 6    | 0.045       | 0.029       | 0.026       | 0.011       | +0.015       |
| 7    | 0.052       | 0.026       | 0.036       | 0.013       | +0.023       |
| 8    | 0.087       | 0.054       | 0.058       | 0.024       | +0.034       |
| 9    | 0.102       | 0.068       | 0.066       | 0.032       | +0.034       |
| 10   | 0.099       | 0.079       | 0.061       | 0.042       | +0.019       |
| 11   | 0.127       | 0.090       | 0.079       | 0.045       | +0.034       |

### Finding: LLM Is Significantly More Selective Than Random — and the Gap Grows

The random baseline's FN rate rises over nodes even with uniform citation, driven purely by the growing pool (more papers compete for the same 30-paper window, so any given paper is shown fewer times, making zero-citation outcomes more likely by chance). But the **LLM's FN rate is consistently and substantially higher** than the random baseline.

By node 11, the LLM ignores 12.7% of papers it was shown, versus 9.0% for random — a **41% higher FN rate** than the random baseline. At the event level, the LLM wastes 7.9% of exposure events (shown but ignored), versus 4.5% for random, a **76% excess** over the baseline.

This is the clearest evidence of genuine LLM selectivity. A random citer, by construction, assigns citations uniformly across whatever it was shown. The LLM systematically discriminates: when it receives a 30-paper context, it preferentially cites a subset and ignores others, regardless of how many times those ignored papers appear in context. This behavior becomes more pronounced as the pool fills with AI-generated papers — the LLM's willingness to ignore a shown paper nearly doubles from node 0 to node 11.

The event-level excess stabilizes around +0.02 to +0.034 from node 5 onward, suggesting the discriminatory behavior reaches a roughly steady-state level rather than spiraling. Still, even at that steady state, the LLM is consistently more exclusive than random.

---

## Metric 3: False Positives (Hallucination Rate — LLM Only)

By construction, the random citer has a 0% hallucination rate — it can only sample from `papers_seen_id`. The LLM can potentially cite papers not in its context window.

| Node | Hallucinated | Total Raw Citations | FP Rate |
|------|-------------|---------------------|---------|
| 0    | 7           | 3,123               | 0.22%   |
| 1    | 3           | 2,960               | 0.10%   |
| 2    | 12          | 2,805               | 0.43%   |
| 3    | 10          | 2,831               | 0.35%   |
| 4    | 14          | 2,744               | 0.51%   |
| 5    | 17          | 2,799               | 0.61%   |
| 6    | 7           | 2,748               | 0.25%   |
| 7    | 12          | 2,734               | 0.44%   |
| 8    | 12          | 2,589               | 0.46%   |
| 9    | 8           | 2,598               | 0.31%   |
| 10   | 11          | 2,647               | 0.42%   |
| 11   | 2           | 2,558               | 0.08%   |
| **Total** | **115** | **32,937**      | **0.35%** |

### Finding: Hallucinations Are Negligible and Show No Trend

The LLM's hallucination rate sits consistently below 0.65% throughout all 12 nodes and shows no upward trend — if anything, node 11 has the lowest rate of all (0.08%). The model remains strongly grounded in its provided reference list regardless of how AI-heavy the pool becomes. With only 115 hallucinated citations out of 32,937, false positives are not a meaningful driver of any observed bias.

This also means the concentration and FN findings are not artifacts of the model citing ghost papers — the citation decisions measured are overwhelmingly about real choices between real papers in the context window.

---

## Global Citation Share: Seed vs. Generated

|           | LLM              | Random           |
|-----------|-----------------|-----------------|
| Seed share of pool | 7.7% (120/1,560) | 7.7% |
| Seed % of all citations | **29.5%** | **27.1%** |
| Avg citations per seed paper | **81.3** | **74.4** |
| Avg citations per generated paper | **18.1** | **18.5** |
| Seed-to-gen citation ratio | **4.50×** | **4.02×** |

Both experiments over-represent seed papers substantially — seed papers are 7.7% of the pool but receive ~28–30% of citations. This again reflects the structural first-mover advantage. But the LLM amplifies the ratio from 4.02× to 4.50× — an additional **12% amplification** of seed over-representation on top of what pure structural dynamics produce.

---

## Synthesis and Conclusion

### What the data shows

Three distinct forces shape citation outcomes in this experiment, and the metrics allow us to disentangle them:

**1. Structural first-mover advantage (dominant force)**
The random baseline demonstrates that most of the Gini growth (0.10 → 0.40) and most of the seed over-representation (~27% of citations) arises purely from pool mechanics — seeds are available in every generation, so they simply accumulate more citation opportunities than late-arriving papers. This is not an LLM effect; it would appear in any citation system with sequential pool growth.

**2. LLM selectivity (real, consistent, moderate)**
The LLM adds a genuine and persistent discrimination effect on top of the structural baseline:
- Gini premium: +0.02 to +0.05 at every node
- Paper-level FN rate: 41% higher than random by node 11
- Event-level FN rate: 76% higher than random by node 11
- Seed citation ratio: 4.50× vs 4.02× random

This is not noise. The LLM, when presented with the same 30 papers as the random citer, consistently passes over a larger fraction of them. It makes qualitative distinctions — likely based on writing quality, abstract clarity, or topic alignment — that the random citer does not. Real papers (seeds) benefit from this because their abstracts are written by humans and are more likely to signal relevance clearly.

**3. Hallucinations (negligible)**
The 0.35% hallucination rate is too small to meaningfully affect any distribution-level metric and shows no upward trend. The LLM is not inventing citations at scale.

### Conclusion

**LLM citation behavior produces genuine but moderate concentration bias beyond what structural dynamics alone explain.** The experiment cannot be summarized as simply "LLMs create citation inequality" — the random baseline makes clear that sequential pool growth would create substantial inequality on its own. However, the LLM's behavior does consistently amplify that baseline inequality across all three measured metrics.

The most meaningful signal is in the false negative rate. Unlike concentration metrics (which are heavily confounded by first-mover structure), the FN rate directly measures the LLM's within-context discrimination. The LLM ignores ~41% more of the papers it was shown than random chance would predict by node 11, and this gap opens up progressively as more AI-generated papers enter the pool. The model appears to apply a quality filter that disfavors AI-generated abstracts — a preference that benefits seed papers and penalizes later-generation work, compounding the structural advantage seeds already hold.

If this dynamic reflects real-world LLM-assisted literature review, the implication is that human-written papers in the seed literature would receive outsized citation amplification, while AI-assisted papers would face both structural disadvantage (arriving late) and behavioral disadvantage (being passed over when shown). The effects are not catastrophic in isolation but compound across generations in a way that meaningfully widens inequality beyond what the underlying research landscape would produce.
