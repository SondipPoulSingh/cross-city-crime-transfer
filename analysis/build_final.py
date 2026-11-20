"""Build the single-file anonymised manuscript (article class, Springer sn-basic reference style).
Tables are inlined, figures are external files Fig1..Fig6 next to the .tex (JGS file naming)."""
import re, pathlib
src = pathlib.Path("manuscript_src.tex").read_text()
src = re.sub(r"\\input\{(tables/[^}]+)\}", lambda m: pathlib.Path(m.group(1)).read_text().rstrip("\n"), src)
a = src.index("\\resizebox{\\textwidth}{!}{%\n\\begin{tikzpicture}"); b = src.index("\\end{tikzpicture}}", a) + len("\\end{tikzpicture}}")
src = src[:a] + "\\includegraphics[width=\\textwidth]{Fig1}" + src[b:]
src = re.sub(r"\\includegraphics\[width=\\textwidth\]\{figures/(Fig\d)\.pdf\}", r"\\includegraphics[width=\\textwidth]{\1}", src)
def rep(x, y):
    global src; assert src.count(x) == 1, x[:60]; src = src.replace(x, y)
rep("\\usepackage{tikz}\\usetikzlibrary{positioning,arrows.meta,fit,calc}\n", "")
rep("\\bibliographystyle{plainnat}", "\\bibliographystyle{sn-basic}")
rep("\\newcommand{\\HMX}{HiST-MAT-X}", "\\newcommand{\\HMX}{HiST-MAT-X}\n\\providecommand{\\bibcommenthead}{}")
rep("% Compiles with pdflatex + bibtex. To move to the Springer Nature template, replace the preamble with\n%   \\documentclass[pdflatex,sn-basic]{sn-jnl}\n% and keep the body unchanged (author-year citations via natbib are supported by sn-basic).",
    "% Single .tex file; compile with pdflatex, bibtex, pdflatex, pdflatex. Figures: Fig1..Fig6 (PDF/EPS) in the same folder.\n% Reference style: Springer Nature basic (sn-basic.bst, author-year).")
for q in ['"BATTERY"', '"ASSAULT 3 \\& RELATED OFFENSES"', '"the share of thefts among this area\'s incidents over the last four weeks"', '"Best target-only"']:
    rep(q, "``" + q[1:-1] + "''")
pathlib.Path("manuscript.tex").write_text(src)
print("ok")
