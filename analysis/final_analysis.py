"""Final cross-session analysis for Paper 2: tables (CSV + LaTeX) and paper figures from the stored outputs."""
import sys, re, json, math
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = Path(sys.argv[1]); OUT = Path(sys.argv[2]); (OUT / "tables").mkdir(parents=True, exist_ok=True); (OUT / "figures").mkdir(exist_ok=True)
r = pd.read_csv(SRC / "tables/results_main.csv", dtype={"budget": str})
a = pd.read_csv(SRC / "tables/results_ablation.csv", dtype={"budget": str})
CITIES = ["CHI", "NYC", "SF"]; CNAME = {"CHI": "Chicago", "NYC": "New York", "SF": "San Francisco"}
BUD = ["4", "8", "16", "52", "all"]; TASKS = ["type", "slot"]; TNAME = {"type": "O1 crime type", "slot": "O2 time of day"}
TO = ["Prior", "HistHeur", "LogReg", "GBDT", "MLP", "TabM-mini", "FTT", "TabICL", "HMX-scratch"]
SHOW = ["Prior", "GBDT", "TabM-mini", "FTT", "TabICL", "GBDT-pooled", "FTT-pre", "HMX-scratch", "HMX-pre"]
W = 26; rng = np.random.default_rng(2026); NB = 2000
IDX = np.stack([np.bincount(rng.integers(0, W, W), minlength=W) for _ in range(NB)]).astype(float); ONE = np.ones((1, W))

def load(table, task, tg, bu, m, s):
    p = SRC / "runs" / (re.sub(r"[^A-Za-z0-9_.-]", "_", f"{table}__{task}|{tg}|{bu}|{m}|{s}") + ".npz")
    return dict(np.load(p)) if p.exists() else None
def stats(table, task, tg, bu, m, seeds=None):
    src = r if table == "main" else a
    ss = seeds if seeds is not None else sorted(src[(src.task == task) & (src.target == tg) & (src.budget == bu) & (src.model == m)].seed.unique())
    return [x for x in (load(table, task, tg, bu, m, s) for s in ss) if x is not None], ss
def ll(S, I, key="nll_w"): return np.mean([(I @ s[key]) / (I @ s["n_w"]) for s in S], 0)
def paired(task, tg, A, B, table="main", key="nll_w"):
    (ma, ba), (mb, bb) = A, B
    sa = set(r[(r.task == task) & (r.target == tg) & (r.budget == ba) & (r.model == ma)].seed) if table == "main" else None
    sb = set(r[(r.task == task) & (r.target == tg) & (r.budget == bb) & (r.model == mb)].seed) if table == "main" else None
    if mb in ("Prior", "HistHeur"): seedsA, seedsB = sorted(sa), [0] * len(sa)
    elif ma == "HMX-zero": seedsA, seedsB = sorted(sa), sorted(sb)
    else: seedsA = seedsB = sorted(sa & sb)
    SA, _ = stats(table, task, tg, ba, ma, seedsA); SB, _ = stats(table, task, tg, bb, mb, seedsB)
    d = ll(SA, IDX, key) - ll(SB, IDX, key); d0 = ll(SA, ONE, key)[0] - ll(SB, ONE, key)[0]
    lo, hi = np.percentile(d, [2.5, 97.5]); return d0, lo, hi, len(SA)
def best_to(task, tg, bu):
    g = r[(r.task == task) & (r.target == tg) & (r.budget == bu)].groupby("model").LL.mean()
    g = g[[m for m in TO if m in g.index]]; return g.idxmin(), g.min()

def tex(df, path, caption, label, bold_min_cols=None):
    cols = list(df.columns)
    s = "\\begin{table}[t]\n\\centering\n\\caption{" + caption + "}\n\\label{" + label + "}\n\\small\n\\begin{tabular}{" + "l" * 2 + "c" * (len(cols) - 2) + "}\n\\toprule\n"
    s += " & ".join(str(c).replace("_", "\\_") for c in cols) + " \\\\\n\\midrule\n"
    for _, row in df.iterrows():
        s += " & ".join(str(v).replace("±", "$\\pm$").replace("_", "\\_") if not str(v).startswith("**") else "\\textbf{" + str(v)[2:].replace("±", "$\\pm$") + "}" for v in row.values) + " \\\\\n"
    s += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"; Path(path).write_text(s)

# ---------- T1/T2: main log-loss tables
for task in TASKS:
    rows = []
    for c in CITIES:
        z = r[(r.task == task) & (r.target == c) & (r.model == "HMX-zero")].LL
        rows.append({"City": CNAME[c], "Weeks": "0", **{m: "–" for m in SHOW}, "HMX-zero": f"{z.mean():.4f} ± {z.std():.4f}"})
        for b in BUD:
            g = r[(r.task == task) & (r.target == c) & (r.budget == b)].groupby("model").LL.agg(["mean", "std"])
            best = g.loc[[m for m in SHOW if m in g.index], "mean"].min(); row = {"City": CNAME[c], "Weeks": b}
            for m in SHOW:
                if m in g.index:
                    v = f"{g.loc[m, 'mean']:.4f} ± {g.loc[m, 'std']:.4f}" if np.isfinite(g.loc[m, "std"]) else f"{g.loc[m, 'mean']:.4f}"
                    row[m] = ("**" if abs(g.loc[m, "mean"] - best) < 5e-5 else "") + v
                else: row[m] = "–"
            row["HMX-zero"] = ""; rows.append(row)
    t = pd.DataFrame(rows); t.to_csv(OUT / f"tables/T{1 if task == 'type' else 2}_main_LL_{task}.csv", index=False)
    tex(t, OUT / f"tables/T{1 if task == 'type' else 2}_main_LL_{task}.tex",
        f"{TNAME[task]}: temperature-scaled test log-loss (mean $\\pm$ s.d. over seeds; lower is better; bold = best). 'all' = full history since 2019. FT-Transformer models use 3 seeds at 52/all, TabICL is limited to 10k context and $\\le$52 weeks.",
        f"tab:main_{task}")

# ---------- T3: effect of pretraining, paired week-block bootstrap
eff = []
for task in TASKS:
    for c in CITIES:
        for b in BUD:
            bm, bv = best_to(task, c, b)
            for lab, A, B in [("FTT-pre − best target-only", ("FTT-pre", b), (bm, b)), ("HMX-pre − best target-only", ("HMX-pre", b), (bm, b)),
                              ("HMX-pre − FTT-pre", ("HMX-pre", b), ("FTT-pre", b)), ("GBDT-pooled − GBDT", ("GBDT-pooled", b), ("GBDT", b)),
                              ("HMX-pre − HMX-scratch", ("HMX-pre", b), ("HMX-scratch", b))]:
                d, lo, hi, n = paired(task, c, A, B)
                eff.append(dict(task=task, city=c, weeks=b, comparison=lab, best_target_only=bm, d=d, lo=lo, hi=hi, n_seeds=n,
                                verdict="better" if hi < 0 else ("worse" if lo > 0 else "tie")))
        bm, _ = best_to(task, c, "4"); d, lo, hi, n = paired(task, c, ("HMX-zero", "0"), (bm, "4"))
        eff.append(dict(task=task, city=c, weeks="0 vs 4", comparison="HMX-zero − best target-only@4wk", best_target_only=bm, d=d, lo=lo, hi=hi, n_seeds=n,
                        verdict="better" if hi < 0 else ("worse" if lo > 0 else "tie")))
eff = pd.DataFrame(eff); eff.to_csv(OUT / "tables/T3_paired_effects.csv", index=False)
summ = eff.assign(short=eff.weeks.isin(["4", "8"])).groupby(["comparison", "task", "short"]).verdict.value_counts().unstack(fill_value=0)
summ.to_csv(OUT / "tables/T3b_paired_effects_summary.csv")
p = eff[eff.comparison.isin(["FTT-pre − best target-only", "HMX-pre − best target-only", "HMX-pre − FTT-pre", "GBDT-pooled − GBDT"])].copy()
p["cell"] = p.apply(lambda x: f"{x.d:+.4f} [{x.lo:+.4f}, {x.hi:+.4f}]" + ("$^{\\dagger}$" if x.verdict == "better" else ""), axis=1)
for task in TASKS:
    tt = p[p.task == task].pivot_table(index=["city", "weeks"], columns="comparison", values="cell", aggfunc="first").reset_index()
    tt["weeks"] = pd.Categorical(tt.weeks, BUD, ordered=True); tt["city"] = pd.Categorical(tt.city, CITIES, ordered=True)
    tt = tt.sort_values(["city", "weeks"]); tt["city"] = tt.city.map(CNAME)
    tex(tt, OUT / f"tables/T3_paired_{task}.tex", f"{TNAME[task]}: paired differences in log-loss with 95\\% week-block bootstrap intervals (negative favours the first model; $\\dagger$ = interval below 0).", f"tab:paired_{task}")

# ---------- T4: secondary metrics at 8 weeks
sec = r[(r.budget == "8") & r.model.isin(SHOW)].groupby(["task", "target", "model"])[["macroF1", "rareF1", "top3", "ECE", "exp_severity", "mistake_severity"]].mean().round(4)
sec.to_csv(OUT / "tables/T4_secondary_8wk.csv")

# ---------- T5: ablations at 8 weeks
VAR = {"no_history": "− history tokens", "no_context": "− cell/neighbour/city context", "type_only_pre": "type-only pretraining (no multi-task)",
       "no_hier": "− hierarchy loss", "no_tokdrop": "− token dropout", "no_area_city": "− area & city tokens", "src_CHI": "source = Chicago only",
       "src_NYC": "source = New York only", "src_SF": "source = San Francisco only"}
ab = []
for task in TASKS:
    for c in CITIES:
        F, _ = stats("abl", task, c, "8", "HMX[full]", [0, 1, 2])
        for v in VAR:
            V, _ = stats("abl", task, c, "8", f"HMX[{v}]", [0, 1, 2])
            if len(V) < 3: continue
            for key, met in [("nll_w", "dLL"), ("esev_w", "dSeverity")]:
                d = ll(V, IDX, key) - ll(F, IDX, key); d0 = ll(V, ONE, key)[0] - ll(F, ONE, key)[0]; lo, hi = np.percentile(d, [2.5, 97.5])
                ab.append(dict(task=task, city=c, variant=v, label=VAR[v], metric=met, d=d0, lo=lo, hi=hi, sig=(lo > 0) or (hi < 0)))
ab = pd.DataFrame(ab); ab.to_csv(OUT / "tables/T5_ablations.csv", index=False)
q = ab.copy(); q["cell"] = q.apply(lambda x: f"{x.d:+.4f}" + ("*" if x.sig else ""), axis=1)
abt = q.pivot_table(index="label", columns=["task", "metric", "city"], values="cell", aggfunc="first")
abt = abt.reindex([VAR[v] for v in VAR]); abt.to_csv(OUT / "tables/T5_ablations_wide.csv")
flat = abt.copy(); flat.columns = [f"{TNAME[t].split()[0]} {m} {c}" for t, m, c in flat.columns]; flat = flat.reset_index().rename(columns={"label": "Variant"})
flat.insert(1, " ", ""); tex(flat, OUT / "tables/T5_ablations.tex", "Ablations at 8 weeks of target history (variant minus full HiST-MAT-X; 3 seeds; positive = the variant is worse; * = 95\\% week-block bootstrap interval excludes 0). dSeverity = expected mistake severity in the crime-type tree / circular slot distance.", "tab:ablations")

# ---------- T6: runtime
rt = r[r.budget.isin(["8", "all"]) & r.model.isin(SHOW)].groupby(["model", "budget"]).secs.mean().unstack().round(0)
rt.to_csv(OUT / "tables/T6_runtime_seconds_per_run.csv")

# ================= FIGURES
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]; CPAL = dict(zip(CITIES, PAL[:3]))
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5, "legend.fontsize": 8, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6, "savefig.dpi": 300, "lines.linewidth": 2})
WEEKX = {"4": 4, "8": 8, "16": 16, "52": 52}
def wx(c, b): return WEEKX.get(b, float(r[(r.target == c) & (r.budget == "all")].weeks.iloc[0]))

# F1: learning curves
LC = [("GBDT", PAL[0], "-", "o", "GBDT (target only)"), ("TabM-mini", PAL[1], "-", "s", "TabM-mini (target only)"),
      ("FTT-pre", PAL[2], "-", "^", "FT-Transformer, source-pretrained"), ("HMX-pre", PAL[3], "-", "D", "HiST-MAT-X, source-pretrained")]
fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.6))
for i, task in enumerate(TASKS):
    for j, c in enumerate(CITIES):
        ax = axes[i, j]
        for m, col, ls, mk, lab in LC:
            g = r[(r.task == task) & (r.target == c) & (r.model == m) & (r.budget != "0")].groupby("budget").LL.agg(["mean", "std"])
            xs = [wx(c, b) for b in BUD if b in g.index]; ys = [g.loc[b, "mean"] for b in BUD if b in g.index]
            ax.plot(xs, ys, ls, color=col, marker=mk, ms=4.5, label=lab, zorder=3, markeredgecolor="white", markeredgewidth=.8)
        z = r[(r.task == task) & (r.target == c) & (r.model == "HMX-zero")].LL.mean()
        ax.axhline(z, color=MUTED, ls=(0, (4, 3)), lw=1.2, label="HiST-MAT-X zero-shot (no target data)")
        ax.set_xscale("log"); ax.set_xticks([4, 8, 16, 52, wx(c, "all")]); ax.set_xticklabels(["4", "8", "16", "52", "all"])
        ax.minorticks_off(); ax.set_title(f"{CNAME[c]} · {TNAME[task]}", color=INK, loc="left")
        lo = r[(r.task == task) & (r.target == c) & r.model.isin([x[0] for x in LC])].groupby(["model", "budget"]).LL.mean()
        ax.set_ylim(min(lo.min(), z) - .004, max(lo.max(), z) + .006)
        if j == 0: ax.set_ylabel("test log-loss (lower is better)")
        if i == 1: ax.set_xlabel("weeks of target-city history")
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(.5, -.02))
plt.tight_layout(rect=(0, .05, 1, 1)); plt.savefig(OUT / "figures/F1_learning_curves.png", bbox_inches="tight"); plt.savefig(OUT / "figures/F1_learning_curves.pdf", bbox_inches="tight"); plt.close()

# F2: pretraining gain vs history length
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=False)
for ax, task in zip(axes, TASKS):
    e = eff[(eff.task == task) & (eff.comparison == "FTT-pre − best target-only")]
    for k, c in enumerate(CITIES):
        ec = e[e.city == c].set_index("weeks").loc[BUD]; x = np.arange(len(BUD)) + (k - 1) * .18
        ax.errorbar(x, ec.d, yerr=[ec.d - ec.lo, ec.hi - ec.d], fmt="o", color=CPAL[c], ms=5, capsize=2.5, lw=1.4, label=CNAME[c], markeredgecolor="white")
    ax.axhline(0, color=MUTED, lw=1); ax.set_xticks(range(len(BUD))); ax.set_xticklabels(BUD)
    ax.set_xlabel("weeks of target-city history   (below 0 = pretraining helps)"); ax.set_title(TNAME[task], loc="left", color=INK)
axes[0].set_ylabel("Δ log-loss, pretrained − best target-only\n(95% week-block bootstrap)")
axes[1].legend(frameon=False, loc="lower right")
plt.tight_layout(); plt.savefig(OUT / "figures/F2_pretraining_gain.png", bbox_inches="tight"); plt.savefig(OUT / "figures/F2_pretraining_gain.pdf", bbox_inches="tight"); plt.close()

# F3: ablations (crime type, dLL and dSeverity)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
order = list(VAR)[::-1]
for ax, met, title in zip(axes, ["dLL", "dSeverity"], ["Δ log-loss", "Δ expected mistake severity"]):
    for k, c in enumerate(CITIES):
        s = ab[(ab.task == "type") & (ab.metric == met) & (ab.city == c)].set_index("variant")
        ys = [i + (k - 1) * .22 for i, v in enumerate(order) if v in s.index]; vs = [v for v in order if v in s.index]
        ax.errorbar(s.loc[vs, "d"], ys, xerr=[s.loc[vs, "d"] - s.loc[vs, "lo"], s.loc[vs, "hi"] - s.loc[vs, "d"]], fmt="o", color=CPAL[c], ms=4.5,
                    capsize=2, lw=1.2, label=CNAME[c], markeredgecolor="white")
    ax.axvline(0, color=MUTED, lw=1); ax.set_title(f"{title}  (O1 crime type, 8 weeks)", loc="left", color=INK); ax.set_xlabel("variant − full model (positive = worse)")
axes[0].set_yticks(range(len(order))); axes[0].set_yticklabels([VAR[v] for v in order]); axes[1].legend(frameon=False, loc="lower right")
plt.tight_layout(); plt.savefig(OUT / "figures/F3_ablations.png", bbox_inches="tight"); plt.savefig(OUT / "figures/F3_ablations.pdf", bbox_inches="tight"); plt.close()

# F4: area-level gain vs volume (8 weeks, crime type, HMX-pre and FTT-pre vs GBDT)
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=True); ar = []
for ax, c in zip(axes, CITIES):
    G, _ = stats("main", "type", c, "8", "GBDT", [0, 1, 2, 3, 4]); Hh, _ = stats("main", "type", c, "8", "HMX-pre", [0, 1, 2, 3, 4])
    n = G[0]["n_a"]; ok = n >= 20
    lg = np.mean([s["nll_a"] / np.maximum(s["n_a"], 1) for s in G], 0); lh = np.mean([s["nll_a"] / np.maximum(s["n_a"], 1) for s in Hh], 0)
    gain = (lg - lh)[ok]; nn = n[ok]
    ax.scatter(nn, gain, s=16, color=CPAL[c], edgecolor="white", linewidth=.6, zorder=3)
    ax.axhline(0, color=MUTED, lw=1); ax.set_xscale("log"); ax.set_title(CNAME[c], loc="left", color=INK); ax.set_xlabel("test incidents in area (log)")
    rho = pd.Series(nn).corr(pd.Series(gain), method="spearman"); ax.text(.97, .92, f"Spearman ρ = {rho:.2f}\n{(gain > 0).mean():.0%} of areas improve",
                                                                           transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=MUTED)
    ar.append(dict(city=c, n_areas=int(ok.sum()), spearman=rho, share_improved=(gain > 0).mean()))
axes[0].set_ylabel("log-loss gain over GBDT\n(HiST-MAT-X, 8 weeks; >0 = better)")
plt.tight_layout(); plt.savefig(OUT / "figures/F4_area_gain.png", bbox_inches="tight"); plt.savefig(OUT / "figures/F4_area_gain.pdf", bbox_inches="tight"); plt.close()
pd.DataFrame(ar).to_csv(OUT / "tables/T7_area_gain.csv", index=False)
for f in ["data_audit.csv", "type_mix_by_city.csv", "slot_mix_by_city.csv", "splits.csv", "label_mapping_audit.csv", "importance.csv"]:
    if (SRC / "tables" / f).exists(): pd.read_csv(SRC / "tables" / f).to_csv(OUT / "tables" / f"A_{f}", index=False)
print("done")
