"""Shared settings of the September 2025 Alaska run comparisons."""

import argparse
from pathlib import Path

ROOT = Path("/work2/noaa/nos-surge/felicioc/STOFS_3D_AK")
OUT_ROOT = ROOT / "comparison_ak_sep25"

# Comparison groups: name -> runs, in plotting (and 2 by 2 panel) order.
GROUPS = {
    "models": ["21", "22", "23", "24"],
    "mesh_dt": ["20", "21", "25", "26"],
}

RUN_LABELS = {
    "20": "M20 (M11 mesh)",
    "21": "M21 (SCHISM)",
    "22": "M22 (UFS SCHISM)",
    "23": "M23 (SCHISM+WWM)",
    "24": "M24 (UFS SCHISM+WW3)",
    "25": "M25 (dt=30 s)",
    "26": "M26 (FINERNORTH)",
}

# Okabe-Ito colorblind-safe colors; unique within each group, obs are black.
RUN_COLORS = {
    "20": "#D55E00",
    "21": "#0072B2",
    "22": "#D55E00",
    "23": "#009E73",
    "24": "#CC79A7",
    "25": "#009E73",
    "26": "#CC79A7",
}


def config_dir(run):
    return ROOT / f"M{run}" / "config"


def load_run_config(run):
    from workflow.core.config import load_config
    return load_config(config_dir(run))


def group_parser(description):
    """Parser with --group: one comparison group, or all of them."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--group", choices=[*GROUPS, "all"], default="all")
    return ap


def selected_groups(args):
    return list(GROUPS) if args.group == "all" else [args.group]


def output_dir(*parts):
    out = OUT_ROOT.joinpath(*parts)
    out.mkdir(parents=True, exist_ok=True)
    return out
