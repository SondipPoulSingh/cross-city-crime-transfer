"""Moran's I (queen, 9,999 permutations) of log test volume per area, from the Session 7 outputs. Usage: python volume_moran.py <spatial_outputs dir>"""
import sys, numpy as np, pandas as pd, geopandas as gpd
SP = sys.argv[1]; UTM = {"CHI": 32616, "NYC": 32618, "SF": 32610}; rng = np.random.default_rng(2026)
ag = pd.read_csv(f"{SP}/area_gains.csv", dtype={"key": str, "weeks": str}); ag = ag[ag.task == "type"]
def key(c, g):
    if c == "CHI": return g["area_numbe"].map(lambda v: str(float(v)))
    if c == "NYC": return g["precinct"].map(lambda v: str(float(v)))
    return g["nhood"].astype(str)
def moran(z, W, nperm=9999):
    keep = W.sum(1) > 0; z, W = z[keep], W[np.ix_(keep, keep)]; W = W / W.sum(1, keepdims=True); z = z - z.mean(); n = len(z)
    I = n / W.sum() * (z @ W @ z) / (z @ z)
    sims = np.array([(lambda p: n / W.sum() * (p @ W @ p) / (p @ p))(rng.permutation(z)) for _ in range(nperm)])
    return I, (1 + np.sum(np.abs(sims - sims.mean()) >= abs(I - sims.mean()))) / (nperm + 1)
rows = []
for c in ["CHI", "NYC", "SF"]:
    g = gpd.read_file(f"{SP}/boundaries_{c}.geojson"); g["key"] = key(c, g).values; g = g.dissolve("key").reset_index().to_crs(UTM[c]).set_index("key")
    d = ag[ag.city == c]; b = [x.buffer(1.0) for x in g.loc[d.key].geometry.values]; n = len(b); W = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if b[i].intersects(b[j]): W[i, j] = W[j, i] = 1
    I, p = moran(np.log(d.n_test.values), W); rows.append(dict(city=c, moran_I_logvolume=I, p_perm=p)); print(c, round(I, 3), p)
pd.DataFrame(rows).to_csv(f"{SP}/moran_volume.csv", index=False)
