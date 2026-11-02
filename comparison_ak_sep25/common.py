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
    "20": "JZ_oldmsh_60s_sch",
    "21": "JZ_newmsh_60s_sch",
    "22": "JZ_newmsh_60s_ufsch",
    "23": "JZ_newmsh_60s_schwwm",
    "24": "JZ_newmsh_60s_ufschww3",
    "25": "JZ_newmsh_30s_sch",
    "26": "FC_newmsh_60s_sch",
}

# Unique within each group, obs are black; M21 keeps its color in both groups.
RUN_COLORS = {
    "20": "#FF00FF",
    "21": "#0000FF",
    "22": "#E60000",
    "23": "#00A000",
    "24": "#FF00FF",
    "25": "#00A000",
    "26": "#E60000",
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
