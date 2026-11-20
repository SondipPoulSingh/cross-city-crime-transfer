"""Journal-format figures (174 mm wide, Liberation Sans = Arial metrics, panel letters, no in-figure titles)."""
import sys, re
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC, TAB, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); OUT.mkdir(parents=True, exist_ok=True)
r = pd.read_csv(SRC / "tables/results_main.csv", dtype={"budget": str})
eff = pd.read_csv(TAB / "T3_paired_effects.csv", dtype={"weeks": str}); ab = pd.read_csv(TAB / "T5_ablations.csv")
CITIES = ["CHI", "NYC", "SF"]; CN = {"CHI": "Chicago", "NYC": "New York", "SF": "San Francisco"}
TN = {"type": "crime type (O1)", "slot": "time of day (O2)"}; BUD = ["4", "8", "16", "52", "all"]
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]; CP = dict(zip(CITIES, PAL[:3])); MK = {"CHI": "o", "NYC": "s", "SF": "^"}
INK, MUT, GRID = "#0b0b0b", "#52514e", "#e6e5e1"; MM = 1 / 25.4
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.edgecolor": MUT, "axes.labelcolor": INK, "xtick.color": MUT,
                     "ytick.color": MUT, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": .5, "lines.linewidth": 1.5, "axes.linewidth": .6, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600})
def letter(ax, s, extra=""): ax.set_title(f"({s}) {extra}", loc="left", color=INK, fontsize=8)
def save(fig, name):
    for ext in ["pdf", "eps", "png"]: fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
def wx(c, b): return {"4": 4, "8": 8, "16": 16, "52": 52}.get(b, float(r[(r.target == c) & (r.budget == "all")].weeks.iloc[0]))

# Fig 2: learning curves
LC = [("GBDT", PAL[0], (0, ()), "o", "GBDT, target only"), ("TabM-mini", PAL[1], (0, (5, 1.5)), "s", "TabM-mini, target only"),
      ("FTT-pre", PAL[2], (0, ()), "^", "FT-Transformer, source-pretrained"), ("HMX-pre", PAL[3], (0, ()), "D", "HiST-MAT-X, source-pretrained")]
fig, axes = plt.subplots(2, 3, figsize=(174 * MM, 112 * MM)); k = 0
for i, task in enumerate(["type", "slot"]):
    for j, c in enumerate(CITIES):
        ax = axes[i, j]
        for m, col, ls, mk, lab in LC:
            g = r[(r.task == task) & (r.target == c) & (r.model == m) & (r.budget != "0")].groupby("budget").LL.mean()
            ax.plot([wx(c, b) for b in BUD], [g[b] for b in BUD], color=col, ls=ls, marker=mk, ms=3.6, label=lab, zorder=3,
                    markeredgecolor="white", markeredgewidth=.6)
        z = r[(r.task == task) & (r.target == c) & (r.model == "HMX-zero")].LL.mean()
        ax.axhline(z, color=MUT, ls=(0, (3, 2)), lw=1, label="HiST-MAT-X, zero-shot (no target data)")
        ax.set_xscale("log"); ax.set_xticks([wx(c, b) for b in BUD]); ax.set_xticklabels(BUD); ax.minorticks_off()
        letter(ax, "abcdef"[k], f"{CN[c]}, {TN[task]}"); k += 1
        if j == 0: ax.set_ylabel("test log-loss")
        if i == 1: ax.set_xlabel("weeks of target-city history")
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.5, -.04))
fig.tight_layout(rect=(0, .07, 1, 1)); save(fig, "Fig2")

# Fig 3: pretraining gain
fig, axes = plt.subplots(1, 2, figsize=(174 * MM, 62 * MM))
for ax, task, s in zip(axes, ["type", "slot"], "ab"):
    e = eff[(eff.task == task) & (eff.comparison == "FTT-pre − best target-only")]
    for k, c in enumerate(CITIES):
        ec = e[e.city == c].set_index("weeks").loc[BUD]; x = np.arange(len(BUD)) + (k - 1) * .2
        ax.errorbar(x, ec.d, yerr=[ec.d - ec.lo, ec.hi - ec.d], fmt=MK[c], color=CP[c], ms=4, capsize=2, lw=1.1, label=CN[c], markeredgecolor="white", markeredgewidth=.5)
    ax.axhline(0, color=MUT, lw=.8); ax.set_xticks(range(len(BUD))); ax.set_xticklabels(BUD)
    ax.set_xlabel("weeks of target-city history"); letter(ax, s, TN[task])
axes[0].set_ylabel("Δ log-loss vs best target-only\n(negative = pretraining helps)")
axes[1].legend(frameon=False, loc="lower right"); fig.tight_layout(); save(fig, "Fig3")

# Fig 4: ablations
VAR = {"no_history": "− history tokens", "no_context": "− cell/neighbour/city context", "type_only_pre": "type-only pretraining",
       "no_hier": "− hierarchy loss", "no_tokdrop": "− token dropout", "no_area_city": "− area and city tokens",
       "src_CHI": "source: Chicago only", "src_NYC": "source: New York only", "src_SF": "source: San Francisco only"}
order = list(VAR)[::-1]
fig, axes = plt.subplots(1, 2, figsize=(174 * MM, 78 * MM), sharey=True)
for ax, met, s, lab in zip(axes, ["dLL", "dSeverity"], "ab", ["Δ log-loss", "Δ expected mistake severity"]):
    for k, c in enumerate(CITIES):
        q = ab[(ab.task == "type") & (ab.metric == met) & (ab.city == c)].set_index("variant")
        vs = [v for v in order if v in q.index]; ys = [order.index(v) + (k - 1) * .24 for v in vs]
        ax.errorbar(q.loc[vs, "d"], ys, xerr=[q.loc[vs, "d"] - q.loc[vs, "lo"], q.loc[vs, "hi"] - q.loc[vs, "d"]], fmt=MK[c], color=CP[c],
                    ms=3.6, capsize=1.6, lw=1, label=CN[c], markeredgecolor="white", markeredgewidth=.5)
    ax.axvline(0, color=MUT, lw=.8); ax.set_xlabel(f"{lab}, variant − full model"); letter(ax, s, f"{lab}, crime type, 8 weeks")
axes[0].set_yticks(range(len(order))); axes[0].set_yticklabels([VAR[v] for v in order]); axes[1].legend(frameon=False, loc="lower right")
fig.tight_layout(); save(fig, "Fig6")

# Fig 5: area gains
def load(task, tg, bu, m, s):
    p = SRC / "runs" / (re.sub(r"[^A-Za-z0-9_.-]", "_", f"main__{task}|{tg}|{bu}|{m}|{s}") + ".npz"); return dict(np.load(p))
fig, axes = plt.subplots(1, 3, figsize=(174 * MM, 58 * MM), sharey=True)
for ax, c, s in zip(axes, CITIES, "abc"):
    G = [load("type", c, "8", "GBDT", k) for k in range(5)]; H = [load("type", c, "8", "HMX-pre", k) for k in range(5)]
    n = G[0]["n_a"]; ok = n >= 20
    gain = (np.mean([g["nll_a"] / np.maximum(g["n_a"], 1) for g in G], 0) - np.mean([h["nll_a"] / np.maximum(h["n_a"], 1) for h in H], 0))[ok]
    ax.scatter(n[ok], gain, s=10, marker=MK[c], color=CP[c], edgecolor="white", linewidth=.4, zorder=3)
    ax.axhline(0, color=MUT, lw=.8); ax.set_xscale("log"); ax.set_xlabel("test incidents in area"); letter(ax, s, CN[c])
    rho = pd.Series(n[ok]).corr(pd.Series(gain), method="spearman")
    ax.text(.97, .95, f"ρ = {rho:.2f}\n{(gain > 0).mean():.0%} of areas improve", transform=ax.transAxes, ha="right", va="top", fontsize=7, color=MUT)
axes[0].set_ylabel("log-loss gain over GBDT\n(positive = HiST-MAT-X better)"); fig.tight_layout(); save(fig, "Fig4")
print("figures written")
