import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import spearmanr

os.makedirs("figures", exist_ok=True)

# ── Load CSVs ──────────────────────────────────────────────────────────────
bfv1  = pd.read_csv("results/phase3_bfv_dataset1.csv")
bfv2  = pd.read_csv("results/phase3_bfv_dataset2.csv")
ckks1 = pd.read_csv("results/phase3_ckks_dataset1.csv")
ckks2 = pd.read_csv("results/phase3_ckks_dataset2.csv")
all_data = pd.concat([bfv1, bfv2, ckks1, ckks2], ignore_index=True)


# ============================================================
# FIGURE 4 — TOTAL LATENCY BREAKDOWN
# ============================================================

sub = all_data[
    (all_data["dataset"] == "dataset1") &
    (all_data["poly_mod_degree"] == 8192)
].groupby(["scheme", "samples"]).agg(
    enc     = ("enc_latency_ms",  "mean"),
    enc_std = ("enc_latency_ms",  "std"),
    exe     = ("exec_latency_ms", "mean"),
    exe_std = ("exec_latency_ms", "std"),
    dec     = ("dec_latency_ms",  "mean"),
    dec_std = ("dec_latency_ms",  "std"),
).reset_index()

cohort_labels = ["n=100", "n=400", "n=801"]
samples_list  = [100, 400, 801]

C_ENC  = "#1a6faf"
C_EXEC = "#f4a23c"
C_DEC  = "#b0d4a0"

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, scheme in zip(axes, ["BFV", "CKKS"]):
    s = sub[sub["scheme"] == scheme].sort_values("samples")

    enc_vals = s["enc"].values
    exe_vals = s["exe"].values
    dec_vals = s["dec"].values
    total_std = np.sqrt(s["enc_std"]**2 + s["exe_std"]**2 + s["dec_std"]**2).values

    x = np.arange(len(samples_list))
    width = 0.55

    ax.bar(x, enc_vals, width, color=C_ENC)
    ax.bar(x, exe_vals, width, bottom=enc_vals, color=C_EXEC)
    ax.bar(x, dec_vals, width, bottom=enc_vals + exe_vals, color=C_DEC)

    # Error bars on total
    totals = enc_vals + exe_vals + dec_vals
    ax.errorbar(x, totals, yerr=total_std,
                fmt="none", color="#333333", capsize=4, linewidth=1.3, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(cohort_labels)
    ax.set_xlabel("Sample count ($n$)", fontsize=10)
    ax.set_ylabel("Total latency (ms)", fontsize=10)
    ax.set_yscale("log")
    ax.set_title(f"{scheme} ($N=8192$)", fontsize=11, fontweight="bold")
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

patches = [
    mpatches.Patch(color=C_ENC,  label="Encryption"),
    mpatches.Patch(color=C_EXEC, label="Execution"),
    mpatches.Patch(color=C_DEC,  label="Decryption"),
]

fig.legend(handles=patches, loc="upper center", ncol=3,
           fontsize=10, bbox_to_anchor=(0.5, 1.02))

plt.tight_layout()
plt.savefig("figures/plot4_total_latency_fixed.png", dpi=600, bbox_inches="tight")
plt.close()
print("Saved: figures/plot4_total_latency_fixed.png")


# ============================================================
# FIGURE 6 — PLAINTEXT VS ENCRYPTED DE SCORE SCATTER
# ============================================================

SCORE_FILES = {
    "plain": "plain_scores_d1_801.npy",
    "bfv":   "bfv_scores_d1_801.npy",
    "ckks":  "ckks_scores_d1_801.npy",
}

missing = [k for k, f in SCORE_FILES.items() if not os.path.exists(f)]

if missing:
    print(
        f"\nSkipping plot6_scatter.png — missing score files: "
        f"{[SCORE_FILES[k] for k in missing]}\n"
        f"Add np.save() calls to your experiment scripts to generate these.\n"
    )
else:
    plain_scores     = np.load(SCORE_FILES["plain"])
    enc_scores_bfv   = np.load(SCORE_FILES["bfv"])
    enc_scores_ckks  = np.load(SCORE_FILES["ckks"])

    rho_bfv,  _ = spearmanr(plain_scores, enc_scores_bfv)
    rho_ckks, _ = spearmanr(plain_scores, enc_scores_ckks)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    configs = [
        (enc_scores_bfv,  "BFV ($N=8192$)",  "#2166ac", rho_bfv,  axes[0]),
        (enc_scores_ckks, "CKKS ($N=8192$)", "#d6604d", rho_ckks, axes[1]),
    ]

    for enc, label, color, rho, ax in configs:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(plain_scores),
                         size=min(5000, len(plain_scores)), replace=False)

        ax.scatter(plain_scores[idx], enc[idx],
                   alpha=0.2, s=3, color=color, rasterized=True)

        lim_min = min(plain_scores.min(), enc.min())
        lim_max = max(plain_scores.max(), enc.max())
        ax.plot([lim_min, lim_max], [lim_min, lim_max],
                "k--", linewidth=1.0, label="$y = x$")

        ax.text(0.05, 0.92, f"$\\rho = {rho:.6f}$",
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="gray", alpha=0.9))

        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)
        ax.set_xlabel("Plaintext DE score", fontsize=10)
        ax.set_ylabel("Encrypted DE score", fontsize=10)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig("figures/plot6_scatter.png", dpi=600, bbox_inches="tight")
    plt.close()
    print("Saved: figures/plot6_scatter.png")