"""Fig. 6: choropleth of area-level gains (journal style), from the Session 7 spatial outputs."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

SP, OUT = Path(sys.argv[1]), Path(sys.argv[2])
CITIES = ["CHI", "NYC", "SF"]; CN = {"CHI": "Chicago", "NYC": "New York", "SF": "San Francisco"}
UTM = {"CHI": 32616, "NYC": 32618, "SF": 32610}
INK, MUT = "#0b0b0b", "#52514e"; MM = 1 / 25.4
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 8, "axes.titlesize": 8, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600})
CM = LinearSegmentedColormap.from_list("div", ["#eb6834", "#f7f7f5", "#2a78d6"])
ag = pd.read_csv(SP / "area_gains.csv", dtype={"key": str, "weeks": str})
mo = pd.read_csv(SP / "moran.csv", dtype={"weeks": str})


def key(c, g):
    if c == "CHI": return g["area_numbe"].map(lambda v: str(float(v)))
    if c == "NYC": return g["precinct"].map(lambda v: str(float(v)))
    return g["nhood"].astype(str)


sel = ag[(ag.task == "type") & (ag.weeks == "8") & (ag.model == "HMX-pre")]
lim = 0.10
fig, axes = plt.subplots(1, 3, figsize=(174 * MM, 80 * MM), gridspec_kw=dict(width_ratios=[1, 1.1, 1]))
for ax, c, s in zip(axes, CITIES, "abc"):
    g = gpd.read_file(SP / f"boundaries_{c}.geojson"); g["key"] = key(c, g).values
    g = g.dissolve("key").reset_index().to_crs(UTM[c])
    d = sel[sel.city == c].set_index("key")
    g["gain"] = g.key.map(d.gain)
    g[g.gain.isna()].plot(ax=ax, color="#e6e5e1", hatch="//////", edgecolor="#9a9994", linewidth=.2)
    g[g.gain.notna()].plot(ax=ax, column="gain", cmap=CM, norm=TwoSlopeNorm(0, -lim, lim), edgecolor=MUT, linewidth=.25)
    r = mo[(mo.city == c) & (mo.task == "type") & (mo.weeks == "8") & (mo.model_A == "HMX-pre") & (mo.weights == "queen")].set_index("variable")
    ax.set_axis_off(); ax.set_aspect("equal")
    ax.text(0, 1.0, f"({s}) {CN[c]}", transform=ax.transAxes, fontsize=8, color=INK, va="bottom")
    f2 = lambda v: f"{v:.2f}".replace("-0.00", "0.00").replace("-", "\u2212")
    ax.text(0, -0.02, f"Moran's I: gain {f2(r.loc['gain', 'moran_I'])} (p = {r.loc['gain', 'p_perm']:.2f})\n"
                      f"after volume {f2(r.loc['gain_resid_logvol', 'moran_I'])} (p = {r.loc['gain_resid_logvol', 'p_perm']:.2f})",
            transform=ax.transAxes, fontsize=7, color=MUT, va="top")
sm = plt.cm.ScalarMappable(cmap=CM, norm=TwoSlopeNorm(0, -lim, lim))
fig.subplots_adjust(left=.01, right=.99, top=.93, bottom=.30, wspace=.08)
cax = fig.add_axes([.25, .13, .5, .03])
cb = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="both", ticks=[-.1, -.05, 0, .05, .1])
cb.ax.tick_params(labelsize=7, color=MUT); cb.outline.set_linewidth(.4)
cb.set_label("log-loss gain over GBDT (blue: HiST-MAT-X better; orange: worse)", fontsize=7, color=INK)
fig.text(.99, .13, "hatched: fewer than 20 test incidents", ha="right", fontsize=6.5, color=MUT)
for ext in ["pdf", "eps", "png"]: fig.savefig(OUT / f"Fig5.{ext}")
print("Fig5 (maps) written; values outside ±0.10:", int((sel.gain.abs() > lim).sum()))
