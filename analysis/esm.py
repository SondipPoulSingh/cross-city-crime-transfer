"""Online Resource 1 (ESM) LaTeX source from the result tables."""
import sys
from pathlib import Path
import numpy as np, pandas as pd

SRC, PAPER = Path(sys.argv[1]), Path(sys.argv[2])
T = SRC / "tables"
r = pd.read_csv(T / "results_main.csv", dtype={"budget": str})
CITIES = ["CHI", "NYC", "SF"]; CN = {"CHI": "Chicago", "NYC": "New York", "SF": "San Francisco"}
BUD = ["0", "4", "8", "16", "52", "all"]
MODELS = ["Prior", "HistHeur", "LogReg", "GBDT", "MLP", "TabM-mini", "FTT", "TabICL", "HMX-scratch",
          "GBDT-pooled", "FTT-pre", "HMX-pre", "HMX-zero"]
MN = {"HistHeur": "Hist", "GBDT-pooled": "GBDT-pool", "HMX-scratch": "HMX-scratch"}
METRICS = [("LL", "test log-loss after temperature scaling (lower is better)", 4, "min"),
           ("macroF1", "macro-F1 (higher is better)", 3, "max"),
           ("rareF1", "rare-type F1, mean F1 of the three least frequent types in the target training data (higher is better)", 3, "max"),
           ("top3", "top-3 accuracy (higher is better)", 3, "max"),
           ("ECE", "expected calibration error, 15 bins (lower is better)", 3, "min"),
           ("exp_severity", "expected mistake severity (lower is better)", 3, "min")]
TASKN = {"type": "O1 crime type", "slot": "O2 time of day"}
out = []
A = out.append


def esc(s):
    return str(s).replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("#", "\\#").replace("$", "\\$")


def table(task, met, desc, dec, best):
    g = r[r.task == task].groupby(["target", "budget", "model"])[met].agg(["mean", "std", "count"])
    A("\\begin{table}[p]\\centering\\scriptsize")
    A(f"\\caption{{{TASKN[task]}: {desc}. Mean $\\pm$ standard deviation over seeds (five seeds; three for TabICL and for FT-Transformer models at 52 weeks and full history; Prior and Hist are deterministic). Bold: best mean in the column within the city. A dash means not run}}")
    A("\\setlength{\\tabcolsep}{3pt}\\begin{tabular}{ll" + "c" * len(BUD) + "}\\toprule")
    A("City & Model & " + " & ".join("0 (zero-shot)" if b == "0" else ("full" if b == "all" else f"{b} wk") for b in BUD) + "\\\\\\midrule")
    for ci, c in enumerate(CITIES):
        bestv = {}
        for b in BUD:
            vals = [g.loc[(c, b, m), "mean"] for m in MODELS if (c, b, m) in g.index and not np.isnan(g.loc[(c, b, m), "mean"])]
            if vals: bestv[b] = (min if best == "min" else max)(vals)
        for mi, m in enumerate(MODELS):
            cells = []
            for b in BUD:
                if (c, b, m) not in g.index or np.isnan(g.loc[(c, b, m), "mean"]):
                    cells.append("--"); continue
                mu, sd, n = g.loc[(c, b, m)]
                s = f"{mu:.{dec}f}" + (f" $\\pm$ {sd:.{dec}f}" if n > 1 else "")
                if round(mu, dec) == round(bestv[b], dec) and b != "0": s = "\\textbf{" + s + "}"
                cells.append(s)
            lead = f"\\multirow{{{len(MODELS)}}}{{*}}{{{c}}}" if mi == 0 else ""
            A(f"{lead} & {MN.get(m, m)} & " + " & ".join(cells) + "\\\\")
        if ci < 2: A("\\midrule")
    A("\\bottomrule\\end{tabular}\\end{table}\n")


A(r"""\documentclass[11pt]{article}
\usepackage[a4paper,margin=2.2cm]{geometry}
\usepackage{mathptmx}\usepackage{booktabs}\usepackage{multirow}\usepackage{longtable}\usepackage{graphicx}\usepackage{caption}\usepackage{url}
\usepackage[T1]{fontenc}
\captionsetup{font=small,labelfont=bf}
\renewcommand{\thetable}{S\arabic{table}}
\begin{document}
\begin{center}{\Large\bfseries Online Resource 1}\\[4pt]
{\large When does cross-city transfer help? A three-city benchmark for incident-level crime prediction under short local histories}\\[4pt]
Journal of Geographical Systems\end{center}
\noindent This resource contains the full result tables for every model, metric, city and history budget (Tables S1--S12), paired comparisons for time of day (S13), the data multipliers (S14), the zero-shot comparison (S15), per-area results (S16), training time (S17), Friedman ranks (S18), token-masking importance (S19--S20), all spatial autocorrelation tests (S21--S22), the raw-to-harmonised label mapping (S23) and the results of the second test window (S24--S27). Abbreviations: CHI Chicago, NYC New York, SF San Francisco; wk weeks of target history; full = all target history since January 2019. Model names follow Section 5.1 of the paper. Machine-readable versions of all tables and per-run results are provided as Online Resource 2.
\clearpage
""")
for task in ["type", "slot"]:
    for met, desc, dec, best in METRICS:
        if task == "slot" and met == "rareF1":
            desc = "rare-slot F1, mean F1 of the three least frequent time slots in the target training data (higher is better)"
        table(task, met, desc, dec, best)
    A("\\clearpage")

# S13 O2 paired
A(r"""\begin{table}[p]\centering\scriptsize
\caption{O2 time of day: paired differences in test log-loss, A $-$ B, with 95\% week-block bootstrap intervals (2,000 replicates). \textbf{Bold}: A significantly better; \textit{italic}: A significantly worse. ``Best target-only'' is the target-only model with the lowest mean log-loss in that row; HMX-scr is HiST-MAT-X trained from scratch on the target city}
\setlength{\tabcolsep}{2.5pt}\begin{tabular}{lllcccc}\toprule
City & Wk & Best target-only & FTT-pre $-$ best & HMX-pre $-$ best & HMX-pre $-$ FTT-pre & GBDT-pool $-$ GBDT \\\midrule""")
A((PAPER / "tables/tab_paired_slot.tex").read_text())
A("\\bottomrule\\end{tabular}\\end{table}\n")

# S14 data multiplier
dm = pd.read_csv(T / "data_multiplier.csv", dtype={"weeks": str})
A("\\begin{table}[p]\\centering\\small\\caption{Data multiplier: weeks of target history the best target-only model would need, by log-linear interpolation of its learning curve, to reach the log-loss of fine-tuned HiST-MAT-X at the given budget. Multiplier = weeks needed / weeks available; values below 1 mean that HiST-MAT-X is worse than the best target-only model at that budget. No multiplier is defined for the full history, where no longer target history exists}")
A("\\begin{tabular}{llrrrrr}\\toprule Task & City & Weeks & HMX-pre LL & Best target-only LL & Weeks needed & Multiplier\\\\\\midrule")
for _, x in dm.iterrows():
    full = x.weeks not in ("4", "8", "16", "52")
    wn = "--" if (pd.isna(x.weeks_needed_by_target_only) or full) else f"{x.weeks_needed_by_target_only:.1f}"
    mu = "--" if full else esc(x.multiplier)
    wk = f"full ({x.weeks})" if full else x.weeks
    A(f"{TASKN[x.task].split()[0]} & {x.target} & {wk} & {x.HMX_LL:.4f} & {x.best_target_only_LL_same_weeks:.4f} & {wn} & {mu}\\\\")
A("\\bottomrule\\end{tabular}\\end{table}\n")

# S15 zero-shot
z = pd.read_csv(T / "zero_shot.csv")
A("\\begin{table}[p]\\centering\\small\\caption{Zero-shot prediction (HiST-MAT-X applied without any target training) against the class prior and the best target-only model with four weeks of history. Significance of the difference to the four-week model is given in the paper (Section 6.5)}")
A("\\begin{tabular}{lllrrr}\\toprule Task & City & Best 4-wk model & Prior LL & HMX-zero LL & Best 4-wk LL\\\\\\midrule")
for _, x in z.iterrows():
    A(f"{TASKN[x.task].split()[0]} & {x.target} & {MN.get(x.model, x.model)} & {x.prior_LL:.4f} & {x.HMX_zero_LL:.4f} & {x.LL:.4f}\\\\")
A("\\bottomrule\\end{tabular}\\end{table}\n")

# S16 area disparity
ad = pd.read_csv(T / "area_disparity.csv")
A("\\begin{table}[p]\\centering\\small\\caption{Area-level gains of HiST-MAT-X over GBDT (crime type, eight weeks; areas with at least 20 test incidents). Gain = GBDT log-loss minus HiST-MAT-X log-loss, averaged over five seeds. Fifths are formed by the number of test incidents per area}")
A("\\begin{tabular}{lrrrrr}\\toprule City & Areas & Gain, lowest fifth & Gain, highest fifth & Spearman $\\rho$ (volume, gain) & Areas improved\\\\\\midrule")
for _, x in ad.iterrows():
    A(f"{CN[x.target]} & {int(x.n_areas)} & {x.gain_lowest_fifth:.4f} & {x.gain_highest_fifth:.4f} & {x.spearman_volume_gain:.2f} & {100*x.areas_improved:.0f}\\%\\\\")
A("\\bottomrule\\end{tabular}\\end{table}\n")

# S17 runtime
rt = pd.read_csv(PAPER.parent / "final/tables/T6_runtime_seconds_per_run.csv")
A("\\begin{table}[p]\\centering\\small\\caption{Mean wall-clock training time per run in seconds (averaged over cities, tasks and seeds), excluding source pretraining, which is shared across budgets and seeds. HiST-MAT-X serves both tasks with one fine-tuned model, whereas the task-specific models need one run per task (Section 6.4 of the paper)}")
A("\\begin{tabular}{lrr}\\toprule Model & 8 weeks & Full history\\\\\\midrule")
for _, x in rt.iterrows():
    A(f"{MN.get(x.model, x.model)} & {x['8']:.0f} & {'--' if pd.isna(x['all']) else format(x['all'], '.0f')}\\\\")
A("\\bottomrule\\end{tabular}\\end{table}\n")

# S18 Friedman
fr = pd.read_csv(T / "friedman_ranks.csv")
A(f"\\begin{{table}}[p]\\centering\\small\\caption{{Average rank of each model by mean test log-loss over the 30 city--task--budget cells with budgets from four weeks to the full history (Dem\\v{{s}}ar 2006). Only models run in every cell are ranked, so TabICL (not run on the full history) and HMX-zero (budget 0 only) are omitted. Friedman test $p = {fr.friedman_p.iloc[0]:.1e}$; Nemenyi critical difference at $\\alpha=0.05$: {fr.CD.iloc[0]:.2f}}}")
A("\\begin{tabular}{lr}\\toprule Model & Average rank\\\\\\midrule")
for _, x in fr.iterrows():
    A(f"{MN.get(x.model, x.model)} & {x.avg_rank:.2f}\\\\")
A("\\bottomrule\\end{tabular}\\end{table}\n")

# S19 importance
TOK = {"wd": "weekday", "month": "month", "hol": "holiday", "prem": "premise group", "area": "area", "city": "city",
       "slot": "time slot", "type": "crime type", "a4t": "area, 4 wk, type mix", "a52t": "area, 52 wk, type mix",
       "a4s": "area, 4 wk, slot mix", "a52s": "area, 52 wk, slot mix", "c52t": "cell, 52 wk, type mix",
       "n4t": "neighbours, 4 wk, type mix", "ci4t": "city, 4 wk, type mix", "ci4s": "city, 4 wk, slot mix", "vol": "volume"}
im = pd.read_csv(T / "importance.csv", dtype={"budget": str})
for bud in ["8", "all"]:
    q = im[im.budget == bud]
    A(f"\\begin{{table}}[p]\\centering\\small\\caption{{Token-masking importance of fine-tuned HiST-MAT-X, {'eight weeks' if bud == '8' else 'full history'}: increase in test log-loss when one input token is replaced by [MASK] at test time (seed 0). Larger values mean the token carries more information. A dash marks the token that is being predicted}}")
    A("\\begin{tabular}{l" + "r" * 6 + "}\\toprule & \\multicolumn{3}{c}{O1 crime type} & \\multicolumn{3}{c}{O2 time of day}\\\\\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}")
    A("Token & CHI & NYC & SF & CHI & NYC & SF\\\\\\midrule")
    for t in TOK:
        cells = []
        for task in ["type", "slot"]:
            for c in CITIES:
                v = q[(q.task == task) & (q.target == c) & (q.token == t)].dLL
                cells.append("--" if len(v) == 0 else f"{v.iloc[0]:+.4f}".replace("-", "$-$").replace("+", "$+$"))
        A(f"{TOK[t]} & " + " & ".join(cells) + "\\\\")
    A("\\bottomrule\\end{tabular}\\end{table}\n")

# S21-S22 spatial autocorrelation
SPD = Path(sys.argv[3]) if len(sys.argv) > 3 else None
if SPD:
    mo = pd.read_csv(SPD / "moran.csv", dtype={"weeks": str})
    A("\\begin{table}[p]\\centering\\small\\caption{Global Moran's $I$ of the area-level gain of model A over GBDT for every comparison (areas with at least 20 test incidents; gain = GBDT log-loss minus model A log-loss, mean over seeds). Two-sided permutation $p$-values, 9,999 permutations. ``After volume'': residuals of a regression of the gain on log test volume. Queen: contiguity weights from official boundaries; kNN: five nearest area centroids. Row-standardised weights. Exploratory analysis added after the main experiments}")
    A("\\footnotesize\\setlength{\\tabcolsep}{3pt}\\begin{tabular}{lllcccc}\\toprule & & & \\multicolumn{2}{c}{Queen} & \\multicolumn{2}{c}{kNN (5)}\\\\\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}")
    A("City & Comparison & Areas & Gain & After volume & Gain & After volume\\\\\\midrule")
    LBL = [("type", "8", "HMX-pre", "O1, 8 wk, HMX-pre (main)"), ("type", "4", "HMX-pre", "O1, 4 wk, HMX-pre"), ("type", "8", "FTT-pre", "O1, 8 wk, FTT-pre"),
           ("slot", "8", "HMX-pre", "O2, 8 wk, HMX-pre"), ("type", "52", "HMX-pre", "O1, 52 wk, HMX-pre")]
    def f(r):
        v = f"{r.moran_I:.3f}".replace("-0.000", "0.000").replace("-", "$-$"); v = f"\\textbf{{{v}}}" if r.p_perm < .05 else v
        return f"{v} ({r.p_perm:.3f})"
    for ci, c in enumerate(CITIES):
        for li, (t, w, mA, lab) in enumerate(LBL):
            q = mo[(mo.city == c) & (mo.task == t) & (mo.weeks == w) & (mo.model_A == mA)]
            g = lambda wt, v: f(q[(q.weights == wt) & (q.variable == v)].iloc[0])
            na = int(q[q.weights == "knn5"].n_areas.iloc[0])
            A(f"{c if li == 0 else ''} & {lab} & {na} & {g('queen', 'gain')} & {g('queen', 'gain_resid_logvol')} & {g('knn5', 'gain')} & {g('knn5', 'gain_resid_logvol')}\\\\")
        if ci < 2: A("\\midrule")
    A("\\bottomrule\\end{tabular}\\par\\smallskip\\footnotesize Bold: $p < 0.05$. In San Francisco one island area has no contiguous neighbour and is omitted from the queen-contiguity tests.\\end{table}\n")
    mv = pd.read_csv(SPD / "moran_volume.csv")
    A("\\begin{table}[p]\\centering\\small\\caption{Global Moran's $I$ of log test volume per area (queen contiguity, 9,999 permutations; same areas as in Table S21)}")
    A("\\begin{tabular}{lcc}\\toprule City & Moran's $I$ & $p$\\\\\\midrule")
    for _, x in mv.iterrows(): A(f"{CN[x.city]} & {x.moran_I_logvolume:.3f} & {x.p_perm:.4f}\\\\")
    A("\\bottomrule\\end{tabular}\\end{table}\n")

# S23 label mapping
lm = pd.read_csv(T / "label_mapping_audit.csv").fillna("")
A("\\clearpage\\scriptsize\\setlength{\\tabcolsep}{3pt}")
A("\\begin{longtable}{llp{9.5cm}l}\\caption{Raw-to-harmonised label mapping. Field: crime type (offence description) or premise. An empty harmonised value means the record was excluded (types) or treated as unknown (premises). San Francisco publishes no premise}\\\\\\toprule City & Field & Raw value & Harmonised\\\\\\midrule\\endfirsthead")
A("\\toprule City & Field & Raw value & Harmonised\\\\\\midrule\\endhead\\bottomrule\\endfoot")
for _, x in lm.sort_values(["city", "field", "mapped", "raw"]).iterrows():
    A(f"{x.city} & {x.field} & {esc(x.raw)} & {esc(x.mapped) if x.mapped else '(excluded)' if x.field == 'type' else '(unknown)'}\\\\")
A("\\end{longtable}\n\\normalsize")
# S24-S26 second test window
W2D = Path(sys.argv[4]) if len(sys.argv) > 4 else None; W2R = Path(sys.argv[5]) if len(sys.argv) > 5 else None
if W2D and W2R:
    r2 = pd.read_csv(W2D / "tables/results_main.csv", dtype={"budget": str})
    M2 = ["Prior", "HistHeur", "LogReg", "GBDT", "TabM-mini", "FTT", "TabICL", "HMX-scratch", "GBDT-pooled", "FTT-pre", "HMX-pre", "HMX-zero"]
    B2 = ["0", "4", "8", "16"]
    for task in ["type", "slot"]:
        g = r2[r2.task == task].groupby(["target", "budget", "model"]).LL.agg(["mean", "std", "count"])
        A("\\clearpage\\begin{table}[p]\\centering\\scriptsize")
        A(f"\\caption{{Second test window (16 June -- 14 December 2025), {TASKN[task]}: test log-loss after temperature scaling, mean $\\pm$ standard deviation over three seeds (Prior and Hist are deterministic). Bold: best mean in the column within the city}}")
        A("\\setlength{\\tabcolsep}{4pt}\\begin{tabular}{llcccc}\\toprule City & Model & 0 (zero-shot) & 4 wk & 8 wk & 16 wk\\\\\\midrule")
        for ci, c in enumerate(CITIES):
            bestv = {b: min(g.loc[(c, b, m), "mean"] for m in M2 if (c, b, m) in g.index) for b in B2 if any((c, b, m) in g.index for m in M2)}
            for mi, m in enumerate(M2):
                cells = []
                for b in B2:
                    if (c, b, m) not in g.index: cells.append("--"); continue
                    mu, sd, n = g.loc[(c, b, m)]; v = f"{mu:.4f}" + (f" $\\pm$ {sd:.4f}" if n > 1 else "")
                    cells.append(f"\\textbf{{{v}}}" if b != "0" and round(mu, 4) == round(bestv[b], 4) else v)
                A(f"{f'\\multirow{{{len(M2)}}}{{*}}{{{c}}}' if mi == 0 else ''} & {MN.get(m, m)} & " + " & ".join(cells) + "\\\\")
            if ci < 2: A("\\midrule")
        A("\\bottomrule\\end{tabular}\\end{table}\n")
    E2 = pd.read_csv(W2R / "w2_paired_effects.csv", dtype={"weeks": str})
    A("\\begin{table}[p]\\centering\\scriptsize\\caption{Paired differences in test log-loss (A $-$ B) with 95\\% week-block bootstrap intervals in both test windows, budgets 4, 8 and 16 weeks. ``Best'' is the best target-only model of the cell (Prior, Hist, LogReg, GBDT, TabM-mini, FTT, TabICL, HMX-scratch), chosen by mean log-loss. Window 1: main experiment (five seeds); window 2: robustness check (three seeds)}")
    A("\\setlength{\\tabcolsep}{2.5pt}\\begin{tabular}{llllccc}\\toprule Comparison & Task & City & Window & 4 wk & 8 wk & 16 wk\\\\\\midrule")
    CMPS = [("FTT-pre - best target-only", "FTT-pre $-$ best"), ("HMX-pre - best target-only", "HMX-pre $-$ best"), ("HMX-pre - FTT-pre", "HMX-pre $-$ FTT-pre"),
            ("GBDT-pooled - GBDT", "GBDT-pool $-$ GBDT"), ("HMX-pre - HMX-scratch", "HMX-pre $-$ HMX-scr")]
    def fc(x):
        f = lambda v: f"{v:+.4f}".replace("-", "$-$").replace("+", "$+$")
        s_ = f"{f(x.d)} [{f(x.lo)}, {f(x.hi)}]"
        return f"\\textbf{{{s_}}}" if x.verdict == "better" else (f"\\textit{{{s_}}}" if x.verdict == "worse" else s_)
    for ki, (k, lab) in enumerate(CMPS):
        first = True
        for task in ["type", "slot"]:
            for c in CITIES:
                for w in ["window 1", "window 2"]:
                    q = E2[(E2.comparison == k) & (E2.task == task) & (E2.city == c) & (E2.window == w)]
                    cells = [fc(q[q.weeks == b].iloc[0]) if len(q[q.weeks == b]) else "--" for b in ["4", "8", "16"]]
                    A(f"{lab if first else ''} & {'O1' if task == 'type' else 'O2'} & {c} & {w[-1]} & " + " & ".join(cells) + "\\\\"); first = False
        if ki < len(CMPS) - 1: A("\\midrule")
    A("\\bottomrule\\end{tabular}\\par\\smallskip\\footnotesize Bold: A significantly better; italic: A significantly worse.\\end{table}\n")
    z = E2[E2.comparison == "HMX-zero - best target-only@4"]
    A("\\begin{table}[p]\\centering\\small\\caption{Zero-shot HiST-MAT-X minus the best target-only model with four weeks of history, both windows (95\\% week-block bootstrap intervals)}")
    A("\\begin{tabular}{lllc}\\toprule Task & City & Window & Difference [95\\% interval]\\\\\\midrule")
    for _, x in z.sort_values(["task", "city", "window"], ascending=[False, True, True]).iterrows():
        A(f"{'O1' if x.task == 'type' else 'O2'} & {x.city} & {x.window[-1]} & {fc(x)}\\\\")
    A("\\bottomrule\\end{tabular}\\end{table}\n")
A("\\clearpage\\section*{References}\\noindent Dem\\v{s}ar J (2006) Statistical comparisons of classifiers over multiple data sets. Journal of Machine Learning Research 7:1--30\n\n\\noindent Moran PAP (1950) Notes on continuous stochastic phenomena. Biometrika 37(1--2):17--23. \\url{https://doi.org/10.1093/biomet/37.1-2.17}\n\\end{document}")
(PAPER / "esm").mkdir(exist_ok=True)
(PAPER / "esm/Online_Resource_1.tex").write_text("\n".join(out))
print("ok")
