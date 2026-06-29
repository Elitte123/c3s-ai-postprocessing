from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

IN = Path("lead_specific_rf_results.csv")
OUTDIR = Path("figures_t2m_audit")
OUTDIR.mkdir(exist_ok=True)

df = pd.read_csv(IN)

for metric in ["MAE_C", "RMSE_C", "Bias_C", "R2"]:
    plt.figure(figsize=(8, 5))

    for model, g in df.groupby("model"):
        g = g.sort_values("lead_days")
        plt.plot(g["lead_days"], g[metric], marker="o", label=model)

    plt.xlabel("Lead time (days)")
    plt.ylabel(metric)
    plt.title(f"Lead-specific model performance: {metric}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out = OUTDIR / f"fig_lead_specific_{metric}.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved:", out)

print("Done.")
