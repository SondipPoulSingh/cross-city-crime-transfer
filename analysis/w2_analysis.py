"""Second test window (Session 8): paired week-block bootstrap effects, side by side with window 1.
Usage: python w2_analysis.py <window-1 outputs dir> <window-2 outputs dir (histmatx_w2)> <out dir>"""
import sys, re
from pathlib import Path
import numpy as np, pandas as pd

W1, W2, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); OUT.mkdir(parents=True, exist_ok=True)
CITIES = ["CHI", "NYC", "SF"]; TASKS = ["type", "slot"]; BUD = ["4", "8", "16"]
TO = ["Prior", "HistHeur", "LogReg", "GBDT", "TabM-mini", "FTT", "TabICL", "HMX-scratch"]   # same target-only set in both windows
NB = 2000


class Win:
    def __init__(self, src):
        self.src = src; self.r = pd.read_csv(src / "tables/results_main.csv", dtype={"budget": str})
        any_run = next((src / "runs").glob("main__*.npz")); self.W = len(np.load(any_run)["n_w"])
        rng = np.random.default_rng(2026)
        self.IDX = np.stack([np.bincount(rng.integers(0, self.W, self.W), minlength=self.W) for _ in range(NB)]).astype(float)
        self.ONE = np.ones((1, self.W))

    def load(self, task, tg, bu, m, s):
        p = self.src / "runs" / (re.sub(r"[^A-Za-z0-9_.-]", "_", f"main__{task}|{tg}|{bu}|{m}|{s}") + ".npz")
        return dict(np.load(p)) if p.exists() else None

    def seeds(self, task, tg, bu, m):
        return set(self.r[(self.r.task == task) & (self.r.target == tg) & (self.r.budget == bu) & (self.r.model == m)].seed)

    def ll(self, S, I): return np.mean([(I @ s["nll_w"]) / (I @ s["n_w"]) for s in S], 0)

    def paired(self, task, tg, A, B):
        (ma, ba), (mb, bb) = A, B; sa, sb = self.seeds(task, tg, ba, ma), self.seeds(task, tg, bb, mb)
        if mb in ("Prior", "HistHeur"): seedsA, seedsB = sorted(sa), [0] * len(sa)
        elif ma == "HMX-zero": seedsA, seedsB = sorted(sa), sorted(sb)
        else: seedsA = seedsB = sorted(sa & sb)
        SA = [x for x in (self.load(task, tg, ba, ma, s) for s in seedsA) if x is not None]
        SB = [x for x in (self.load(task, tg, bb, mb, s) for s in seedsB) if x is not None]
        if not SA or not SB: return np.nan, np.nan, np.nan, 0
        d = self.ll(SA, self.IDX) - self.ll(SB, self.IDX); d0 = self.ll(SA, self.ONE)[0] - self.ll(SB, self.ONE)[0]
        lo, hi = np.percentile(d, [2.5, 97.5]); return d0, lo, hi, len(SA)

    def best_to(self, task, tg, bu, seeds=None):
        q = self.r[(self.r.task == task) & (self.r.target == tg) & (self.r.budget == bu)]
        if seeds is not None: q = q[q.seed.isin(seeds) | q.model.isin(["Prior", "HistHeur"])]
        g = q.groupby("model").LL.mean(); g = g[[m for m in TO if m in g.index]]; return g.idxmin()


rows = []
for wname, src in [("window 1", W1), ("window 2", W2)]:
    w = Win(src)
    for task in TASKS:
        for c in CITIES:
            if c not in set(w.r.target): continue
            for b in BUD:
                if not len(w.seeds(task, c, b, "FTT-pre")): continue
                bm = w.best_to(task, c, b)
                for lab, A, B in [("FTT-pre - best target-only", ("FTT-pre", b), (bm, b)), ("HMX-pre - best target-only", ("HMX-pre", b), (bm, b)),
                                  ("HMX-pre - FTT-pre", ("HMX-pre", b), ("FTT-pre", b)), ("GBDT-pooled - GBDT", ("GBDT-pooled", b), ("GBDT", b)),
                                  ("HMX-pre - HMX-scratch", ("HMX-pre", b), ("HMX-scratch", b))]:
                    d, lo, hi, n = w.paired(task, c, A, B)
                    rows.append(dict(window=wname, task=task, city=c, weeks=b, comparison=lab, best_target_only=bm, d=d, lo=lo, hi=hi, n_seeds=n,
                                     verdict="better" if hi < 0 else ("worse" if lo > 0 else "tie")))
            if len(w.seeds(task, c, "0", "HMX-zero")):
                bm = w.best_to(task, c, "4"); d, lo, hi, n = w.paired(task, c, ("HMX-zero", "0"), (bm, "4"))
                rows.append(dict(window=wname, task=task, city=c, weeks="0 vs 4", comparison="HMX-zero - best target-only@4", best_target_only=bm,
                                 d=d, lo=lo, hi=hi, n_seeds=n, verdict="better" if hi < 0 else ("worse" if lo > 0 else "tie")))
E = pd.DataFrame(rows); E.to_csv(OUT / "w2_paired_effects.csv", index=False)
S = E.groupby(["comparison", "window", "weeks"]).verdict.value_counts().unstack(fill_value=0)
S.to_csv(OUT / "w2_summary.csv"); print(S.to_string())
mx = E[E.comparison == "FTT-pre - best target-only"].pivot_table(index=["task", "city", "weeks"], columns="window", values="d").round(4)
print(mx.to_string()); mx.to_csv(OUT / "w2_ftt_gain_by_window.csv")
