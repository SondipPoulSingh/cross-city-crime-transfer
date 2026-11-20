"""Main-text table and numbers for the second test window. Usage: python w2_tables.py <w2_res dir> <paper dir>"""
import sys, json
from pathlib import Path
import pandas as pd
RES, PAPER = Path(sys.argv[1]), Path(sys.argv[2])
E = pd.read_csv(RES / "w2_paired_effects.csv", dtype={"weeks": str})
CN = {"CHI": "Chicago", "NYC": "New York", "SF": "San Francisco"}; TN = {"type": "O1 type", "slot": "O2 time"}
def cell(q):
    if not len(q): return "--"
    x = q.iloc[0]; s = f"{x.d:+.3f}".replace("+0.000", "0.000").replace("-0.000", "0.000").replace("-", "$-$").replace("+", "$+$")
    return f"\\textbf{{{s}}}" if x.verdict == "better" else (f"\\textit{{{s}}}" if x.verdict == "worse" else s)
rows = []
for task in ["type", "slot"]:
    for ci, c in enumerate(["CHI", "NYC", "SF"]):
        for mi, (m, lab) in enumerate([("FTT-pre - best target-only", "FTT-pre"), ("HMX-pre - best target-only", "HMX-pre")]):
            cells = [cell(E[(E.window == w) & (E.task == task) & (E.city == c) & (E.weeks == b) & (E.comparison == m)])
                     for w in ["window 1", "window 2"] for b in ["4", "8", "16"]]
            lead = (f"\\multirow{{6}}{{*}}{{{TN[task]}}}" if ci == 0 and mi == 0 else "") + " & " + (f"\\multirow{{2}}{{*}}{{{CN[c]}}}" if mi == 0 else "")
            rows.append(f"{lead} & {lab} & " + " & ".join(cells) + " \\\\")
        if ci < 2: rows.append("\\cmidrule(lr){2-9}")
    if task == "type": rows.append("\\midrule")
(PAPER / "tables/tab_window2.tex").write_text("\n".join(rows) + "\n")
S = {}
for comp in E.comparison.unique():
    for w in ["window 1", "window 2"]:
        for b in E.weeks.unique():
            q = E[(E.comparison == comp) & (E.window == w) & (E.weeks == b)]
            if len(q): S[f"{comp}|{w}|{b}"] = q.verdict.value_counts().to_dict()
json.dump(S, open(RES / "w2_counts.json", "w"), indent=1)
for k, v in S.items(): print(k, v)
