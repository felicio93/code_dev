#!/usr/bin/env python3
"""
Station time series of several runs against the observations (black), for all
stations of station.in. Reads, matches and scores the data with the workflow's
own station_skill and wave_skill functions, so a run's numbers here equal those
of its individual figures. A variable is drawn for the runs that have it.

    python compare_stations.py [--group models|mesh_dt|all]

Output: <OUT_ROOT>/stations/<group>/{station}_{variable}.jpg, skill_metrics.csv
"""

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from workflow.core.station_parser import parse_station_in
from workflow.models.schism.postprocess import station_skill as ss
from workflow.models.schism_wwm.postprocess import wave_skill as wwm
from workflow.models.ufs_schism_ww3.postprocess import wave_skill as ww3

from common import (RUN_COLORS, RUN_LABELS, group_parser, load_run_config,
                    output_dir, selected_groups, GROUPS)

COLUMNS = ["station_id", "name", "source", "variable", "depth_m", "run",
           "n_points", "mean_obs", "bias", "rmse", "r2", "start", "end"]


class Run:
    """One run's config and its cached station output frames."""

    def __init__(self, run_id):
        self.id = run_id
        self.cfg = load_run_config(run_id)
        self.start, self.end = ss._skill_window(self.cfg)
        self.rule = ss._resample_rule(self.cfg)
        self.stations = parse_station_in(
            Path(self.cfg["project_dir"]) / f"M{run_id}" / "fix" / "station.in")
        self._frames = {}
        self.wave_index = None
        self.model_type = str(self.cfg.get("model_type", "schism")).lower()

    def frame(self, staout_num):
        if staout_num not in self._frames:
            self._frames[staout_num] = ss._load_model_staout(
                self.cfg, staout_num, self.stations, self.start, self.end)
        return self._frames[staout_num]

    def series(self, staout_num, line_index):
        frame = self.frame(staout_num)
        if frame.empty or line_index not in frame.columns:
            return None
        return frame[line_index]

    def wave_series(self, station_id, token):
        """Series of one wave variable, or None for a run without waves."""
        if self.model_type == "schism_wwm":
            if self.wave_index is None:
                self.wave_index = wwm._build_station_index_map(self.cfg)
            col = self.wave_index.get(station_id)
            if col is None:
                return None
            series = wwm._load_wwm_station(
                self.cfg, station_id, col, wwm.WAVE_VAR_PLAN[token][1],
                self.start, self.end)
        elif self.model_type == "ufs_schism_ww3":
            series = ww3._load_ww3_station(
                self.cfg, station_id, ww3.WAVE_VAR_PLAN[token][1],
                self.start, self.end)
        else:
            return None
        return None if series.empty else series


def plot_comparison(obs, models, title, ylabel, out_jpg):
    """The individual figures' layout; the legend sits below the axes because
    it holds one metrics entry per run. Observations are the black line."""
    fig, ax = plt.subplots(figsize=(15, 3.8))
    ax.plot(obs.index, obs.values, color="k", linewidth=1)
    for run, series, label in models:
        ax.plot(series.index, series.values, label=label,
                color=RUN_COLORS[run], linewidth=1.3)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", color="gray", linewidth=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2,
              fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_jpg, dpi=150)
    plt.close(fig)


def score(run, series, obs, unit, dlabel, metrics_fn, row, rows, rule=None):
    """Metrics of one run; returns the (run, series, legend label) to plot."""
    res = metrics_fn(obs, series, rule or run.rule)
    if res is None:
        return None
    n, mean_obs, bias, rmse, r2 = res
    rows.append(dict(row, run=f"M{run.id}", n_points=n, mean_obs=mean_obs,
                     bias=bias, rmse=rmse, r2=r2, start=run.start, end=run.end))
    tag = f" [{dlabel}]" if dlabel else ""
    return (run.id, series,
            f"{RUN_LABELS[run.id]}{tag} [R²: {r2:.2f}; "
            f"RMSE: {rmse:.2f} {unit}; Bias: {bias:.2f} {unit}]")


def compare_physical(runs, st, source, lookup, out_dir, rows):
    """All station_skill variables of one station."""
    ref = runs[0]
    sid = st["station_id"]
    ndbc_filter = ss._ndbc_token_filter(ref.cfg)
    done = set()
    for tok in st["vars"]:
        T = tok.strip().upper()
        if source == "NDBC" and (T in ("WL", "ELEV")
                                 or (ndbc_filter and T not in ndbc_filter)):
            continue
        entry = ss._station_for_var(T, sid, lookup)
        if entry is None:
            continue
        name = ss._clean_name(entry["name"])
        depth = entry["depth"]
        dlabel = ss._depth_label(depth)
        got = ss._obs_for_variable(source, tok, ref.cfg, sid,
                                   ref.start, ref.end)
        if got is None:
            continue
        kind, payload = got
        base = dict(station_id=sid, name=name, source=source, depth_m=depth)

        if kind == "wind":
            if "wind" in done:
                continue
            done.add("wind")
            for comp, staout, key in ss.WIND_COMPONENTS:
                obs = payload[key]
                row = dict(base, variable=comp)
                models = []
                for run in runs:
                    series = run.series(staout, entry["line_index"])
                    if series is None:
                        continue
                    item = score(run, series, obs, "m/s", dlabel, ss._metrics,
                                 row, rows)
                    if item:
                        models.append(item)
                if models:
                    plot_comparison(
                        obs, models,
                        f"{source} ({sid}): {name} — Wind {comp} (m/s)",
                        f"{comp} (m/s)", out_dir / f"{sid}_{comp}.jpg")
            continue

        plan = ss.VAR_PLAN.get(T)
        if plan is None or plan["product"] in done:
            continue
        done.add(plan["product"])
        row = dict(base, variable=plan["product"])
        models = []
        for run in runs:
            series = run.series(plan["staout"], entry["line_index"])
            if series is None:
                continue
            if T in ss._AIR_PRESSURE_TOKENS:
                series = series / 100.0
            item = score(run, series, payload, plan["unit"], dlabel,
                         ss._metrics, row, rows)
            if item:
                models.append(item)
        if models:
            plot_comparison(
                payload, models,
                f"{source} ({sid}): {name} — {plan['label']} [{dlabel}]",
                f"{plan['label']} ({plan['unit']})",
                out_dir / f"{sid}_{plan['product']}.jpg")


def compare_waves(runs, st, out_dir, rows):
    """NDBC wave variables of one station, for the runs with a wave model."""
    wave_runs = [r for r in runs
                 if r.model_type in ("schism_wwm", "ufs_schism_ww3")]
    if not wave_runs:
        return
    ref = wave_runs[0]
    sid = st["station_id"]
    name = re.sub(r"\s+z=[-\d.]+m$", "", st["name"]).strip()
    done = set()
    for tok in st["vars"]:
        T = tok.strip().upper()
        if T not in wwm.WAVE_TOKENS:
            continue
        canonical = "TM01" if T == "TM02" else T
        if canonical in done:
            continue
        done.add(canonical)
        ndbc_col, _, label, unit, _ = wwm.WAVE_VAR_PLAN[T]
        obs = wwm._load_ndbc_wave_var(ref.cfg, sid, ndbc_col, ref.start, ref.end)
        if obs.empty:
            continue
        row = dict(station_id=sid, name=name, source="NDBC",
                   variable=canonical, depth_m=0.0)

        def metrics(o, m, rule):
            return wwm._metrics(o, m, rule, circular=(unit == "deg"))

        models = []
        for run in wave_runs:
            series = run.wave_series(sid, T)
            if series is None:
                continue
            item = score(run, series, obs, unit, "", metrics, row, rows)
            if item:
                models.append(item)
        if models:
            plot_comparison(obs, models,
                            f"NDBC ({sid}): {name} — {label}",
                            f"{label} ({unit})",
                            out_dir / f"{sid}_{canonical.lower()}.jpg")


def compare_group(group):
    print(f"\n=== {group}: M{', M'.join(GROUPS[group])}", flush=True)
    runs = [Run(r) for r in GROUPS[group]]
    ref = runs[0]
    out_dir = output_dir("stations", group)
    lookup = ss._build_station_lookup(ref.stations)

    seen, stations = set(), []
    for s in ref.stations:
        if s["depth"] == 0.0 and s["station_id"] not in seen:
            seen.add(s["station_id"])
            stations.append(s)

    rows = []
    for st in stations:
        print(f"  {st['source']} {st['station_id']}", flush=True)
        compare_physical(runs, st, st["source"], lookup, out_dir, rows)
        if st["source"] == "NDBC":
            compare_waves(runs, st, out_dir, rows)

    csv = out_dir / "skill_metrics.csv"
    pd.DataFrame(rows, columns=COLUMNS).to_csv(csv, index=False)
    print(f"  {len(rows)} skill rows -> {csv}", flush=True)


def main():
    args = group_parser(__doc__.split("\n\n")[0]).parse_args()
    for group in selected_groups(args):
        compare_group(group)


if __name__ == "__main__":
    main()
