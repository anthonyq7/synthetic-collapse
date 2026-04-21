import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


SCRIPT_DIR = Path(__file__).resolve().parent

ALPHA = 0.05
M = 12
ALPHA_BONF = ALPHA / M

PAIRS: List[Tuple[str, str]] = [
    ("L1", "L2"),
    ("L1", "L1P"),
]

CONCENTRATION_PATH = SCRIPT_DIR / "concentration_by_node.jsonl"

HEADER_COLOR = "#4472C4"
ALT_ROW_COLOR = "#D9E2F3"
REJECT_COLOR = "#70AD47"
FAIL_COLOR = "#C0392B"


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def format_pvalue(p: float) -> str:
    if not np.isfinite(p):
        return "N/A"
    if p < 0.0001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def z_test_right_unequal(
    p1: float, n1: int, p2: float, n2: int
) -> Tuple[float, float]:
    """Right-tailed two-proportion z-test (Ha: p1 > p2) with unequal sample sizes."""
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    p_hat = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = (p_hat * (1 - p_hat) * (1 / n1 + 1 / n2)) ** 0.5
    if se == 0:
        return float("nan"), float("nan")
    z = (p1 - p2) / se
    p_value = float(1 - stats.norm.cdf(z))
    return z, p_value


def column_labels(cond_a: str, cond_b: str) -> List[str]:
    return [
        "Node",
        f"{cond_a} Top-10% Share",
        f"{cond_b} Top-10% Share",
        "z",
        "p-value",
        f"alpha (Bonf) = {ALPHA_BONF:.5f}",
        "Decision",
    ]


def output_paths(cond_a: str, cond_b: str) -> Tuple[Path, Path]:
    jsonl_path = SCRIPT_DIR / f"concentration_ztest_{cond_a}_vs_{cond_b}.jsonl"
    table_path = SCRIPT_DIR / f"concentration_ztest_table_{cond_a}_vs_{cond_b}.png"
    return jsonl_path, table_path


def run_tests_for_pair(
    by_cond_node: Dict[Tuple[str, int], Dict],
    cond_a: str,
    cond_b: str,
) -> Tuple[List[Dict], List[List[str]], List[str]]:
    missing = []
    for cond in (cond_a, cond_b):
        for node in range(M):
            if (cond, node) not in by_cond_node:
                missing.append((cond, node))
    if missing:
        raise ValueError(
            f"concentration_by_node.jsonl is missing rows for pair "
            f"({cond_a} vs {cond_b}): {missing}"
        )

    result_rows: List[Dict] = []
    cell_rows: List[List[str]] = []
    decisions: List[str] = []

    for node in range(M):
        a = by_cond_node[(cond_a, node)]
        b = by_cond_node[(cond_b, node)]

        p1 = a["top_10pct_share"] / 100.0
        n1 = a["total_citations"]
        p2 = b["top_10pct_share"] / 100.0
        n2 = b["total_citations"]

        z, p_value = z_test_right_unequal(p1, n1, p2, n2)

        if np.isfinite(p_value) and p_value < ALPHA_BONF:
            decision = "Reject H0"
        else:
            decision = "Fail to Reject H0"

        result_rows.append(
            {
                "pair": f"{cond_a}_vs_{cond_b}",
                "condition_a": cond_a,
                "condition_b": cond_b,
                "node": node,
                "p1_pct": round(p1 * 100, 4),
                "n1": n1,
                "p2_pct": round(p2 * 100, 4),
                "n2": n2,
                "z": (round(z, 6) if np.isfinite(z) else None),
                "p_value": (round(p_value, 8) if np.isfinite(p_value) else None),
                "alpha_bonf": ALPHA_BONF,
                "decision": decision,
            }
        )

        cell_rows.append(
            [
                str(node),
                f"{p1 * 100:.2f}%",
                f"{p2 * 100:.2f}%",
                f"{z:.3f}" if np.isfinite(z) else "N/A",
                format_pvalue(p_value),
            ]
        )
        decisions.append(decision)

        print(
            f"[{cond_a} vs {cond_b}][Node {node}] "
            f"{cond_a}={p1 * 100:.2f}% (n={n1}), "
            f"{cond_b}={p2 * 100:.2f}% (n={n2}), "
            f"z={'N/A' if not np.isfinite(z) else f'{z:.3f}'}, "
            f"p={format_pvalue(p_value)}, decision={decision}"
        )

    return result_rows, cell_rows, decisions


def render_table(
    cell_rows: List[List[str]],
    decisions: List[str],
    col_labels: List[str],
    title: str,
    out_path: Path,
) -> None:
    full_rows = [row + [decision] for row, decision in zip(cell_rows, decisions)]
    n_cols = len(col_labels)

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.axis("off")

    table = ax.table(
        cellText=full_rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor(HEADER_COLOR)
            cell.set_text_props(color="white", weight="bold")
        elif col_idx == n_cols - 1:
            decision = decisions[row_idx - 1]
            if decision == "Reject H0":
                cell.set_facecolor(REJECT_COLOR)
            else:
                cell.set_facecolor(FAIL_COLOR)
            cell.set_text_props(color="white", weight="bold")
        elif row_idx % 2 == 0:
            cell.set_facecolor(ALT_ROW_COLOR)

    ax.set_title(title, fontsize=11, fontweight="bold", pad=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


def write_jsonl(result_rows: List[Dict], out_path: Path) -> None:
    with out_path.open("w") as f:
        for row in result_rows:
            f.write(json.dumps(row) + "\n")
    print(f"Saved {out_path}")


def main() -> None:
    if not CONCENTRATION_PATH.exists():
        raise FileNotFoundError(
            f"Missing {CONCENTRATION_PATH}. Run analyze_plurality.py first."
        )

    rows = load_jsonl(CONCENTRATION_PATH)
    by_cond_node: Dict[Tuple[str, int], Dict] = {
        (r["condition"], r["node"]): r for r in rows
    }

    for cond_a, cond_b in PAIRS:
        print(f"\n=== {cond_a} vs {cond_b} ===")
        result_rows, cell_rows, decisions = run_tests_for_pair(
            by_cond_node, cond_a, cond_b
        )
        jsonl_path, table_path = output_paths(cond_a, cond_b)
        write_jsonl(result_rows, jsonl_path)

        title = (
            f"6.4 Plurality: Citation Concentration by Node ({cond_a} vs {cond_b})\n"
            f"Right-tailed two-proportion z-test  |  Ha: p_{cond_a} > p_{cond_b}"
        )
        render_table(
            cell_rows,
            decisions,
            column_labels(cond_a, cond_b),
            title,
            table_path,
        )


if __name__ == "__main__":
    main()
