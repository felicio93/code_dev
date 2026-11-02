#!/usr/bin/env python3
"""
M23 and M24 satellite altimetry figures, side by side.

    python compare_altimetry.py

Output: <OUT_ROOT>/altimetry/{figure}.jpg
"""

from PIL import Image, ImageDraw

from common import ROOT, RUN_LABELS, output_dir
from compare_gifs import font

RUNS = ["23", "24"]
FIGURES = ["altimetry_tracks_by_source", "altimetry_tracks_by_time",
           "altimetry_scatter_time_delta", "altimetry_scatter_dist_delta",
           "altimetry_scatter_depth", "altimetry_scatter_coast_dist",
           "altimetry_scatter_source"]
HEADER = 60     # pixels above each figure for the run label


def main():
    out_dir = output_dir("altimetry")
    for name in FIGURES:
        paths = [ROOT / f"M{r}" / f"P{r}" / f"P{r}_collocate_altimetry"
                 / f"{name}.jpg" for r in RUNS]
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            print(f"  {name}: SKIPPED, missing {', '.join(missing)}")
            continue
        images = [Image.open(p).convert("RGB") for p in paths]
        height = max(i.height for i in images)
        images = [i.resize((round(i.width * height / i.height), height),
                           Image.LANCZOS) for i in images]
        sheet = Image.new("RGB", (sum(i.width for i in images), height + HEADER),
                          "white")
        draw = ImageDraw.Draw(sheet)
        x = 0
        for run, image in zip(RUNS, images):
            sheet.paste(image, (x, HEADER))
            draw.text((x + 20, 12), RUN_LABELS[run], fill="black", font=font(36))
            x += image.width
        out = out_dir / f"{name}.jpg"
        sheet.save(out, quality=92)
        print(f"  {name} -> {out}")


if __name__ == "__main__":
    main()
