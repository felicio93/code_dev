#!/usr/bin/env python3
"""
2 by 2 GIFs of SST, SSS, SSH, X and Y current velocity and vertical velocity
from the frames plot_outputs saved under each run, one frame per day (the
HOUR frame of each day). Uses the days all four runs have.

    python compare_gifs.py [--group models|mesh_dt|all]

Output: <OUT_ROOT>/gifs/<group>/{variable}.gif
"""

from PIL import Image, ImageDraw, ImageFont

from common import (GROUPS, ROOT, RUN_LABELS, group_parser, output_dir,
                    selected_groups)

VARIABLES = ["temperature", "salinity", "elevation", "horizontalVelX",
             "horizontalVelY", "verticalVelocity"]
HOUR = "T00"         # frame time tag used for each day
PANEL_WIDTH = 1000   # pixels of each of the four panels
FPS = 4              # as plot_outputs_fps
LABEL_RIGHT = 150    # pixels of the run label from the panel right edge
LABEL_BOTTOM = 85    # pixels of the run label from the panel bottom edge


def frames_of(run, var):
    """{time tag: frame path} of the daily frames of one variable of one run."""
    fdir = ROOT / f"M{run}" / f"P{run}" / f"P{run}_plot_outputs" / "frames"
    return {p.stem.split("__")[1]: p for p in fdir.glob(f"{var}__*{HOUR}.jpg")}


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:   # Pillow < 10.1
        return ImageFont.load_default()


def composite(paths, labels):
    panels = []
    for path, label in zip(paths, labels):
        img = Image.open(path).convert("RGB")
        height = round(img.height * PANEL_WIDTH / img.width)
        img = img.resize((PANEL_WIDTH, height), Image.LANCZOS)
        draw = ImageDraw.Draw(img)
        x = img.width - LABEL_RIGHT - 18 * len(label)
        y = img.height - LABEL_BOTTOM - 40
        draw.rectangle((x, y, x + 18 * len(label), y + 40), fill="white",
                       outline="black")
        draw.text((x + 6, y + 4), label, fill="black", font=font(28))
        panels.append(img)
    height = max(p.height for p in panels)
    sheet = Image.new("RGB", (2 * PANEL_WIDTH, 2 * height), "white")
    for i, panel in enumerate(panels):
        sheet.paste(panel, ((i % 2) * PANEL_WIDTH, (i // 2) * height))
    return sheet.quantize(colors=256, method=Image.MEDIANCUT,
                          dither=Image.NONE)


def make_gif(group, var):
    runs = GROUPS[group]
    frames = {r: frames_of(r, var) for r in runs}
    empty = [f"M{r}" for r in runs if not frames[r]]
    if empty:
        print(f"  {var}: SKIPPED, no frames for {', '.join(empty)}", flush=True)
        return
    times = sorted(set.intersection(*(set(frames[r]) for r in runs)))
    if not times:
        print(f"  {var}: SKIPPED, the runs share no frame time", flush=True)
        return
    for r in runs:
        extra = len(frames[r]) - len(times)
        if extra:
            print(f"  {var}: M{r} has {extra} frame(s) not used", flush=True)
    labels = [RUN_LABELS[r] for r in runs]
    sheets = (composite([frames[r][t] for r in runs], labels) for t in times)
    first = next(sheets)
    gif = output_dir("gifs", group) / f"{var}.gif"
    first.save(gif, save_all=True, append_images=sheets,
               duration=round(1000 / FPS), loop=0)
    print(f"  {var}: {len(times)} frames, {times[0]} to {times[-1]} -> {gif}",
          flush=True)


def main():
    args = group_parser(__doc__.split("\n\n")[0]).parse_args()
    for group in selected_groups(args):
        print(f"\n=== {group}: M{', M'.join(GROUPS[group])}", flush=True)
        for var in VARIABLES:
            make_gif(group, var)


if __name__ == "__main__":
    main()
