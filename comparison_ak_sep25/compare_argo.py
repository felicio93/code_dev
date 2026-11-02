#!/usr/bin/env python3
"""
Argo mean bias and RMSE against depth, M21, M26 and M25 on the same axes, for
temperature and salinity. Same method as the workflow's argo_profiles figures
(profiles interpolated to 5 m levels), but scored only on the profiles and
levels that all runs have, so no run is judged on extra data.

    python compare_argo.py

Output: <OUT_ROOT>/argo/argo_bias_rmse.jpg, argo_bias_rmse.csv
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from workflow.models.schism.postprocess import argo_plots as ap

from common import RUN_COLORS, RUN_LABELS, load_run_config, output_dir

RUNS = ["21", "26", "25"]
VARIABLES = ["temperature", "salinity"]
STEP = 5.0      # m, as plot_argo_profiles


def load_errors(run, var, common_depth, max_depth):
    """(profile times, model - obs [profile, level]) of one run."""
    cfg = load_run_config(run)
    ds, nc = ap._load_collocated(ap._out_dir(cfg), var)
    if ds is None:
        raise SystemExit(f"M{run}: no collocated {var} file")
    ds = ap._apply_dist_filter(ds, cfg.get("collocate_argo_plot_max_dist_km"))
    meta = ap.VAR_META[var]
    obs = ds[meta["obs_var"]].values
    mod = ds[meta["mod_var"]].values
    dep = ap._depth_pos(ds["depth"].values)
    times = ds["time"].values
    err = np.array([
        ap._interp_profile(dep[i], mod[i], common_depth)
        - ap._interp_profile(dep[i], obs[i], common_depth)
        for i in range(obs.shape[0])])
    print(f"  M{run} {var}: {nc.name}, {obs.shape[0]} profiles", flush=True)
    return times, err


def common_errors(var, common_depth, max_depth):
    """Errors of all runs on their shared profiles and levels."""
    loaded = {r: load_errors(r, var, common_depth, max_depth) for r in RUNS}
    shared = None
    for times, _ in loaded.values():
        unique, counts = np.unique(times, return_counts=True)
        once = unique[counts == 1]
        shared = once if shared is None else np.intersect1d(shared, once)
    errors = {}
    for run, (times, err) in loaded.items():
        errors[run] = err[np.isin(times, shared)][
            np.argsort(times[np.isin(times, shared)])]
    shared_levels = np.all([~np.isnan(errors[r]) for r in RUNS], axis=0)
    for run in RUNS:
        errors[run] = np.where(shared_levels, errors[run], np.nan)
    print(f"  {var}: {len(shared)} shared profiles", flush=True)
    return errors, shared_levels


def main():
    cfg = load_run_config(RUNS[0])
    max_depth = float(cfg.get("collocate_argo_plot_max_depth", 300))
    common_depth = np.arange(0, max_depth + STEP, STEP)
    out_dir = output_dir("argo")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=True,
                             constrained_layout=True)
    rows = []
    for row, var in enumerate(VARIABLES):
        meta = ap.VAR_META[var]
        errors, shared_levels = common_errors(var, common_depth, max_depth)
        n_profiles = shared_levels.sum(axis=0)
        ax_bias, ax_rmse = axes[row]
        for run in RUNS:
            bias = np.nanmean(errors[run], axis=0)
            rmse = np.sqrt(np.nanmean(errors[run] ** 2, axis=0))
            ax_bias.plot(bias, common_depth, color=RUN_COLORS[run], lw=2,
                         label=RUN_LABELS[run])
            ax_rmse.plot(rmse, common_depth, color=RUN_COLORS[run], lw=2,
                         label=RUN_LABELS[run])
            rows.append(pd.DataFrame(dict(
                variable=var, run=f"M{run}", depth_m=common_depth, bias=bias,
                rmse=rmse, n_profiles=n_profiles)))
        ax_bias.axvline(0, color="k", lw=1.2)
        ax_bias.set_title(f"{meta['label']}: Mean Bias (Model − Obs)",
                          fontweight="bold")
        ax_rmse.set_title(f"{meta['label']}: RMSE", fontweight="bold")
        ax_bias.set_xlabel(f"Bias ({meta['unit']})")
        ax_rmse.set_xlabel(f"RMSE ({meta['unit']})")
        ax_bias.set_ylabel("Depth (m)")
        for ax in (ax_bias, ax_rmse):
            ax.grid(True, ls="--", alpha=0.4)
            ax.legend(loc="best")
    axes[0, 0].set_ylim(max_depth, 0)
    fig.suptitle(f"SCHISM vs Argo, shared profiles  |  Depth ≤ "
                 f"{max_depth:.0f} m  |  {cfg['start_date']} to {cfg['end_date']}",
                 fontweight="bold")
    out = out_dir / "argo_bias_rmse.jpg"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    pd.concat(rows).to_csv(out_dir / "argo_bias_rmse.csv", index=False)
    print(f"  -> {out}", flush=True)


if __name__ == "__main__":
    main()
