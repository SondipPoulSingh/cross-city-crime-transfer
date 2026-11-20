# Cross-city pretraining for incident-level crime prediction under short histories

Code and results for the paper "When does cross-city transfer help? A three-city benchmark for incident-level crime prediction under short local histories".

## Data
All data are downloaded automatically from public open-data portals (Socrata APIs):

| City | Incidents | Area boundaries |
|---|---|---|
| Chicago | `ijzp-q8t2` (Crimes – 2001 to Present) | `igwz-8jzy` (community areas) |
| New York | `qgea-i56i` (NYPD Complaint Data Historic) and `5uac-w243` (Current Year To Date) | `y76i-bdw7` (police precincts) |
| San Francisco | `wg3w-h783` (Police Department Incident Reports: 2018 to Present) | `j2bu-swwd` (analysis neighborhoods) |

## How the experiments were run
The grid (3 target cities × 6 history budgets × up to 5 seeds × 13 models × 2 tasks, plus ablations) was run on Kaggle (one NVIDIA T4 GPU) across several 12-hour sessions. Every session resumes from the previous session's output, and runs that have already finished are never repeated.

| Notebook | Role |
|---|---|
| `01_…CrossCity` | Session 1: download, harmonisation, features, Chicago as target |
| `02_…Session2_NYC` | Session 2: protocol v2 (epoch floor), New York |
| `03_…Session3_SF` | Session 3: San Francisco |
| `04_…Session4_NYCall_SFlong` | Session 4: New York full history, San Francisco long budgets |
| `05_…Session5_Ablations` | Session 5: ablations |
| `06_…Session6_CHI_rerun` | Session 6: final pipeline (protocol v3); Chicago runs repeated under v3 |
| `07_…Session7_Spatial_v2` | Session 7 (CPU): area-level spatial autocorrelation (Moran's I) and maps |
| `08_…Session8_Window2` | Session 8: second test window (16 June – 14 December 2025) for budgets 0–16 weeks (robustness check) |

Notebook 06 contains the complete final pipeline. Set `RUN_MODE = "synthetic"` in it to check the code within minutes without internet access.

## Tables and figures
After the sessions have run, use the scripts in `analysis/`:

```
python analysis/final_analysis.py <outputs> final   # summary tables (T1–T7)
python analysis/paper_tables.py <outputs> final/tables paper
python analysis/paper_figs.py   <outputs> final/tables paper/figures
python analysis/paper_fig5_maps.py <spatial_outputs> paper/figures
python analysis/volume_moran.py <spatial_outputs>
python analysis/w2_analysis.py <outputs> <histmatx_w2> <w2_res>
python analysis/w2_tables.py <w2_res> paper
python analysis/esm.py <outputs> paper <spatial_outputs> <histmatx_w2> <w2_res>
python analysis/build_final.py   # run inside paper/: single-file manuscript.tex
```

## Results
`results.zip` contains everything needed to reproduce the tables and figures without re-running the experiments:
`results/window1` (main test window: per-run metrics in `tables/`, per-week and per-area statistics for every run in `runs/*.npz`),
`results/window2` (second test window) and `results/spatial` (Moran's I tests and area-level gains). Unzip it and pass
`results/window1` as `<outputs>`, `results/window2` as `<histmatx_w2>` and `results/spatial` as `<spatial_outputs>` to the scripts above.
Running `final_analysis.py` and `paper_tables.py` on these files reproduces the tables of the paper exactly.
The manuscript source is in `paper/` (`python analysis/build_final.py` inlines the tables).

## Licence
Code: MIT licence.
