"""Build every manuscript table and a numbers file directly from stored results (no hand-copied numbers)."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

SRC, TAB, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); (OUT / "tables").mkdir(parents=True, exist_ok=True)
r = pd.read_csv(SRC / "tables/results_main.csv", dtype={"budget": str})
eff = pd.read_csv(TAB / "T3_paired_effects.csv", dtype={"weeks": str}); ab = pd.read_csv(TAB / "T5_ablations.csv")
audit = pd.read_csv(TAB / "A_data_audit.csv").set_index("city"); splits = pd.read_csv(TAB / "A_splits.csv", dtype={"budget": str})
CITIES = ["CHI", "NYC", "SF"]; CN = {"CHI": "Chicago", "NYC": "New York", "SF": "San Francisco"}; BUD = ["4", "8", "16", "52", "all"]
NAME = {"GBDT": "GBDT", "TabM-mini": "TabM-mini", "FTT": "FTT", "TabICL": "TabICL", "GBDT-pooled": "GBDT-pool", "FTT-pre": "FTT-pre",
        "HMX-scratch": "HMX-scr", "HMX-pre": "HMX-pre", "MLP": "MLP", "LogReg": "LogReg", "Prior": "Prior", "HistHeur": "Hist"}
macros = {}
def M(k, v): macros[k] = v
f4 = lambda x: f"{x:.4f}"; f3 = lambda x: f"{x:.3f}"
def sgn(x, d=4):
    if abs(x) < 0.5 * 10 ** (-d): return f"{0:.{d}f}"
    return f"{x:+.{d}f}".replace("-", "$-$").replace("+", "$+$")

# ---- Table 1: data
rows = []
for c in CITIES:
    a = audit.loc[c]
    rows.append(f"{CN[c]} & {int(a.raw_rows):,} & {int(a.kept_rows):,} & {int(a.n_areas)} & {int(a.n_cells)} & {100*a.unknown_time:.1f} & "
                f"{100*a.unknown_premise:.1f} & {int(a.rows_per_week):,} & {int(a.test_rows):,} \\\\")
    M(f"raw{c}", f"{int(a.raw_rows):,}"); M(f"kept{c}", f"{int(a.kept_rows):,}"); M(f"areas{c}", str(int(a.n_areas))); M(f"perweek{c}", f"{int(a.rows_per_week):,}")
    M(f"test{c}", f"{int(a.test_rows):,}")
(OUT / "tables/tab_data.tex").write_text("\n".join(rows).replace(",", "{,}") + "\n")

# ---- Table: splits (training rows per budget)
rows = []
for c in CITIES:
    s = splits[splits.city == c].set_index("budget")
    rows.append(f"{CN[c]} & " + " & ".join(f"{int(s.loc[b,'train_rows']):,} / {int(s.loc[b,'val_rows']):,}" for b in BUD) + " \\\\")
(OUT / "tables/tab_splits.tex").write_text("\n".join(rows).replace(",", "{,}") + "\n")

# ---- Tables: main log-loss (mean over seeds), compact model set
SHOW = ["GBDT", "TabM-mini", "FTT", "TabICL", "HMX-scratch", "GBDT-pooled", "FTT-pre", "HMX-pre"]
for task in ["type", "slot"]:
    rows = []
    for c in CITIES:
        for bi, b in enumerate(BUD):
            g = r[(r.task == task) & (r.target == c) & (r.budget == b)].groupby("model").LL.mean()
            best = g[[m for m in SHOW if m in g.index]].min()
            cells = []
            for m in SHOW:
                if m not in g.index: cells.append("--")
                else: cells.append(("\\textbf{%s}" if abs(g[m] - best) < 5e-5 else "%s") % f4(g[m]))
            lead = f"\\multirow{{6}}{{*}}{{{CN[c]}}}" if bi == 0 else ""
            rows.append(f"{lead} & {b} & " + " & ".join(cells) + " \\\\")
        z = r[(r.task == task) & (r.target == c) & (r.model == "HMX-zero")].LL
        rows.append(f" & 0 (zero-shot) & \\multicolumn{{8}}{{l}}{{HMX-zero: {f4(z.mean())} (prior at 4 weeks: "
                    f"{f4(r[(r.task==task)&(r.target==c)&(r.budget=='4')&(r.model=='Prior')].LL.mean())})}} \\\\")
        if c != "SF": rows.append("\\midrule")
    (OUT / f"tables/tab_ll_{task}.tex").write_text("\n".join(rows) + "\n")
    sd = r[(r.task == task) & r.model.isin(SHOW) & (r.budget != "0")].groupby(["target", "budget", "model"]).LL.std()
    M(f"sdmax{task}", f4(sd.max())); M(f"sdmed{task}", f4(sd.median()))

# ---- Table: paired effects (FTT-pre and HMX-pre vs best target-only; HMX vs FTT-pre; pooled vs GBDT)
CMP = [("FTT-pre − best target-only", "FTT-pre vs best target-only"), ("HMX-pre − best target-only", "HMX-pre vs best target-only"),
       ("HMX-pre − FTT-pre", "HMX-pre vs FTT-pre"), ("GBDT-pooled − GBDT", "GBDT-pool vs GBDT")]
for task in ["type", "slot"]:
    rows = []
    for c in CITIES:
        for bi, b in enumerate(BUD):
            cells = []
            for key, _ in CMP:
                x = eff[(eff.task == task) & (eff.city == c) & (eff.weeks == b) & (eff.comparison == key)].iloc[0]
                s = f"{sgn(x.d, 3)} [{sgn(x.lo, 3)}, {sgn(x.hi, 3)}]"
                cells.append(f"\\textbf{{{s}}}" if x.verdict == "better" else (f"\\textit{{{s}}}" if x.verdict == "worse" else s))
            lead = f"\\multirow{{5}}{{*}}{{{c}}}" if bi == 0 else ""
            bt = eff[(eff.task == task) & (eff.city == c) & (eff.weeks == b)].best_target_only.iloc[0]
            rows.append(f"{lead} & {b} & {NAME.get(bt, bt)} & " + " & ".join(cells) + " \\\\")
        if c != "SF": rows.append("\\midrule")
    (OUT / f"tables/tab_paired_{task}.tex").write_text("\n".join(rows) + "\n")

# win counts for text
def count(comp, weeks, verdict, task=None):
    e = eff[(eff.comparison == comp) & eff.weeks.isin(weeks)]
    if task: e = e[e.task == task]
    return int((e.verdict == verdict).sum()), len(e)
for comp, tag in [("FTT-pre − best target-only", "ftt"), ("HMX-pre − best target-only", "hmx"), ("HMX-pre − HMX-scratch", "hmxpre"),
                  ("HMX-pre − FTT-pre", "hmxftt"), ("GBDT-pooled − GBDT", "pool")]:
    for wk, wt in [(["4", "8"], "short"), (["16"], "sixteen"), (["52", "all"], "long")]:
        for v in ["better", "tie", "worse"]:
            n, tot = count(comp, wk, v); M(f"{tag}{wt}{v}", str(n)); M(f"{tag}{wt}n", str(tot))
for task in ["type", "slot"]:
    for c in CITIES:
        for b in ["4", "8", "16"]:
            x = eff[(eff.task == task) & (eff.city == c) & (eff.weeks == b) & (eff.comparison == "FTT-pre − best target-only")].iloc[0]
            M(f"gain{task}{c}{b}", f"{-x.d:.3f}"); M(f"gainlo{task}{c}{b}", f"{-x.hi:.3f}"); M(f"gainhi{task}{c}{b}", f"{-x.lo:.3f}")
            y = eff[(eff.task == task) & (eff.city == c) & (eff.weeks == b) & (eff.comparison == "HMX-pre − HMX-scratch")].iloc[0]
            M(f"hmxgain{task}{c}{b}", f"{-y.d:.3f}")
            h = eff[(eff.task == task) & (eff.city == c) & (eff.weeks == b) & (eff.comparison == "HMX-pre − FTT-pre")].iloc[0]
            M(f"hmxftt{task}{c}{b}", f"{h.d:+.4f}")
for task in ["type", "slot"]:
    for c in CITIES:
        x = eff[(eff.task == task) & (eff.city == c) & (eff.comparison == "HMX-zero − best target-only@4wk")].iloc[0]
        M(f"zs{task}{c}", f"{x.d:+.4f}"); M(f"zslo{task}{c}", f"{x.lo:+.4f}"); M(f"zshi{task}{c}", f"{x.hi:+.4f}"); M(f"zsv{task}{c}", x.verdict)
        M(f"zsbest{task}{c}", NAME.get(x.best_target_only, x.best_target_only))
h = eff[eff.comparison == "HMX-pre − FTT-pre"]; M("hmxfttmin", f"{h.d.min():+.4f}"); M("hmxfttmax", f"{h.d.max():+.4f}")
M("hmxfttworsemax", f"{h[h.verdict=='worse'].d.max():.3f}"); M("hmxfttworsemin", f"{h[h.verdict=='worse'].d.min():.3f}")

# ---- Table: secondary metrics at 8 weeks
SEC = ["GBDT", "TabM-mini", "FTT", "TabICL", "GBDT-pooled", "FTT-pre", "HMX-scratch", "HMX-pre"]
rows = []
g = r[r.budget == "8"].groupby(["task", "target", "model"])[["macroF1", "rareF1", "ECE", "exp_severity", "top3"]].mean()
for m in SEC:
    cells = []
    for c in CITIES: cells += [f3(g.loc[("type", c, m), "macroF1"]), f3(g.loc[("type", c, m), "rareF1"])]
    for c in CITIES: cells += [f3(g.loc[("type", c, m), "exp_severity"])]
    for c in CITIES: cells += [f3(g.loc[("type", c, m), "ECE"])]
    rows.append(f"{NAME[m]} & " + " & ".join(cells) + " \\\\")
(OUT / "tables/tab_secondary.tex").write_text("\n".join(rows) + "\n")
for c in CITIES:
    sev = g.loc["type"].xs(c, level="target").loc[SEC].exp_severity; M(f"sevbest{c}", NAME[sev.idxmin()])
    M(f"sevhmx{c}", f3(sev["HMX-pre"])); M(f"sevftt{c}", f3(sev["FTT-pre"]))

# ---- Table: ablations
VAR = [("no_history", "Remove history tokens"), ("no_context", "Remove cell, neighbour and city context"),
       ("type_only_pre", "Pretrain on crime type only"), ("no_hier", "Remove hierarchy loss ($\\epsilon=\\lambda=0$)"),
       ("no_tokdrop", "Remove token dropout"), ("no_area_city", "Remove area and city tokens"),
       ("src_CHI", "Single source: Chicago"), ("src_NYC", "Single source: New York"), ("src_SF", "Single source: San Francisco")]
rows = []
for v, lab in VAR:
    cells = []
    for task, met in [("type", "dLL"), ("type", "dSeverity"), ("slot", "dLL")]:
        for c in CITIES:
            q = ab[(ab.task == task) & (ab.metric == met) & (ab.city == c) & (ab.variant == v)]
            if not len(q): cells.append("--"); continue
            q = q.iloc[0]; s = sgn(q.d); cells.append(s + ("$^{*}$" if q.sig else ""))
    rows.append(f"{lab} & " + " & ".join(cells) + " \\\\")
(OUT / "tables/tab_ablation.tex").write_text("\n".join(rows) + "\n")
for v, _ in VAR:
    for task, met in [("type", "dLL"), ("type", "dSeverity"), ("slot", "dLL")]:
        q = ab[(ab.task == task) & (ab.metric == met) & (ab.variant == v)]
        if len(q): M(f"ab{v.replace('_','')}{task}{met}min", f"{q.d.min():.3f}"); M(f"ab{v.replace('_','')}{task}{met}max", f"{q.d.max():.3f}")

# ---- runtime
rt = r[r.model.isin(["GBDT", "TabM-mini", "FTT-pre", "HMX-pre", "TabICL"])].groupby(["model", "budget"]).secs.mean()
# HMX trains one model for both tasks (time recorded once per task row); FT-Transformer trains one model per task
per = {}
for m in ["GBDT", "TabM-mini", "FTT-pre", "TabICL"]:
    for b in ["8", "all"]:
        if (m, b) in rt.index: per[(m, b)] = 2 * rt[(m, b)]
for b in ["8", "all"]: per[("HMX-pre", b)] = rt[("HMX-pre", b)]
M("rtftt8", f"{per[('FTT-pre','8')]:.0f}"); M("rthmx8", f"{per[('HMX-pre','8')]:.0f}"); M("rtfttall", f"{per[('FTT-pre','all')]:.0f}"); M("rthmxall", f"{per[('HMX-pre','all')]:.0f}")
M("rtratio8", f"{per[('FTT-pre','8')]/per[('HMX-pre','8')]:.1f}"); M("rtratioall", f"{per[('FTT-pre','all')]/per[('HMX-pre','all')]:.1f}")
M("rtgbdt8", f"{per[('GBDT','8')]:.0f}"); M("rtgbdtall", f"{per[('GBDT','all')]:.0f}")

# ---- area
ar = pd.read_csv(TAB / "T7_area_gain.csv").set_index("city")
for c in CITIES: M(f"rho{c}", f"{ar.loc[c,'spearman']:.2f}"); M(f"improved{c}", f"{100*ar.loc[c,'share_improved']:.0f}")
# ---- counts
M("nmain", f"{len(r):,}"); M("nabl", str(len(pd.read_csv(SRC / 'tables/results_ablation.csv'))))
tw = json.load(open(SRC / "logs/test_window.json")); M("testT0", str(tw["T0"])); M("testT1", str(tw["LAST_WEEK"]))

with open(OUT / "numbers.tex", "w") as fh:
    for k, v in macros.items():
        name = "N" + "".join(ch for ch in k if ch.isalpha())
        if any(ch.isdigit() for ch in k):   # digits are not allowed in macro names: spell them
            name = "N" + k.replace("0", "zero").replace("1", "one").replace("2", "two").replace("3", "three").replace("4", "four") \
                .replace("5", "five").replace("6", "six").replace("7", "seven").replace("8", "eight").replace("9", "nine")
            name = "".join(ch for ch in name if ch.isalpha())
        fh.write(f"\\newcommand{{\\{name}}}{{{str(v).replace(',', '{,}')}}}\n")
json.dump(macros, open(OUT / "numbers.json", "w"), indent=0)
print(len(macros), "numbers")
