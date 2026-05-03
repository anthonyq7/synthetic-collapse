import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("./figures", exist_ok=True)

# Exact Proposition 7 node-0 parameters
s0 = 0.25
M = 120

# Exact exchangeability upper bound
R = np.linspace(0.0, 1.0, 1200)
upper = ((1 - s0 * R) ** M - (1 - s0) ** M) / (1 - (1 - s0) ** M)
upper = np.clip(upper, 0, 1)

# Observed node-0 seed results
models = [
    ("GPT-5 mini", 0.324, 0.133),
    ("Claude Haiku 4.5", 0.274, 0.083),
]

# ---------- Plot 1: linear scale ----------
fig1, ax1 = plt.subplots(figsize=(7.5, 5.0), dpi=180)

ax1.plot(R, upper, linestyle="--", linewidth=2.2, label="Exchangeable upper bound (Prop. 7)")
for label, x, y in models:
    ax1.scatter([x], [y], s=50)

# Adjusted text positions (terminology updated)
ax1.text(0.34, 0.16, "GPT-5 mini\nR=0.324, ER=0.133", fontsize=9)
ax1.text(0.10, 0.108, "Claude Haiku 4.5\nR=0.274, ER=0.083", fontsize=9)

# Bigger impossibility text
ax1.fill_between(R, upper, 1, alpha=0.08)
ax1.text(0.4, 0.9, "Infeasible Region under exchangeability", fontsize=13, color = "#1b5e07")

ax1.set_xlabel("Shown-conditioned seed citation rate $R_{seed,0}$")
ax1.set_ylabel("Seed exclusion rate")
ax1.set_title("Node-0 exchangeability impossibility")
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1.02)
ax1.grid(True, alpha=0.2)
ax1.legend(frameon=False)

plt.tight_layout()

linear_png = "./figures/Node-0_Exchangeability_Impossibility_Plot_linear.png"
fig1.savefig(linear_png, bbox_inches="tight")

# ---------- Plot 2: log y-scale ----------
fig2, ax2 = plt.subplots(figsize=(7.5, 5.0), dpi=180)

# Avoid log(0) by flooring only for display
upper_log = np.maximum(upper, 1e-8)

ax2.plot(R, upper_log, linestyle="--", linewidth=2.2, label="Exchangeable upper bound (Prop. 7)")
for label, x, y in models:
    ax2.scatter([x], [y], s=50)

# Adjusted text positions (terminology updated)
ax2.text(0.34, 0.16, "GPT-5 mini\nR=0.324, ER=0.133", fontsize=9)
ax2.text(0.10, 0.11, "Claude Haiku 4.5\nR=0.274, ER=0.083", fontsize=9)

# Shade region above the bound, clipped for log display
ax2.fill_between(R, upper_log, 1, alpha=0.08)
ax2.text(0.4, 0.00015, "Infeasible Region under exchangeability", fontsize=13, color = "#1b5e07")

ax2.set_yscale("log")
ax2.set_xlabel("Shown-conditioned seed citation rate $R_{seed,0}$")
ax2.set_ylabel("Seed exclusion rate (log scale)")
ax2.set_title("Node-0 exchangeability impossibility (log-scale y-axis)")
ax2.set_xlim(0, 1)
ax2.set_ylim(1e-8, 1.02)
ax2.grid(True, alpha=0.2, which="both")
ax2.legend(frameon=False)

plt.tight_layout()

log_png = "./figures/Node-0_Exchangeability_Impossibility_Plot_log.png"

fig2.savefig(log_png, bbox_inches="tight")

plt.show()

print(linear_png)

print(log_png)
