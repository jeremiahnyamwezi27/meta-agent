"""Reproduce the results_graph.png figure."""

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

epochs = list(range(8))

gold_labels = [67, 67, 67, 67, 80, 80, 80, 80]
llm_judge   = [67, 67, 67, 80, 80, 87, 87, 87]
baseline    = 67

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(epochs, gold_labels, "o-", color="#4285F4", label="Gold labels", markersize=6, linewidth=2)
ax.plot(epochs, llm_judge, "o--", color="#DB4437", label="LLM judge (no labels)", markersize=6, linewidth=2)
ax.axhline(y=baseline, color="#B0B0B0", linewidth=1)
ax.text(max(epochs) + 0.15, baseline, f"baseline {baseline}%", va="center", fontsize=8, color="#999")

ax.set_xlabel("Epoch")
ax.set_ylabel("Best holdout accuracy")
ax.set_title("Harness optimization on tau-bench (Haiku 4.5)")
ax.set_ylim(55, 100)
ax.set_xticks(epochs)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend(loc="lower right")
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig("images/results_graph.png", dpi=150)
print("Saved images/results_graph.png")
