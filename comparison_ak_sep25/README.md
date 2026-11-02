# September 2025 Alaska run comparisons

Run on Hercules, in the `swf_plot` environment, with the workflow installed.
Outputs go to `$R/comparison_ak_sep25/` (`ROOT` and groups are in `common.py`).

| Script | Compares | Output |
|---|---|---|
| `compare_stations.py` | all stations: M21-M24 (`models`), M20, M21, M25, M26 (`mesh_dt`); waves for M23, M24 | `stations/<group>/` |
| `compare_gifs.py` | 2 by 2 SST, X and Y current, vertical velocity for both groups | `gifs/<group>/` |
| `compare_altimetry.py` | M23 and M24 altimetry figures side by side | `altimetry/` |
| `compare_argo.py` | M21 and M26 Argo bias and RMSE on shared profiles | `argo/` |

`--group models|mesh_dt|all` selects the group of the first two scripts.
