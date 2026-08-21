# Update: 4 separate jobs (A/B/C/D each with own 8h walltime)

Following up on the previous comment — I re-ran the matrix as **4 independent
SLURM jobs** (one per config, each with its own 8h budget) instead of one
combined job. This confirmed the earlier A/B/C numbers and, most importantly,
**let Config D's MPI run complete for the first time.**

Code versions unchanged: OCSMesh `felicio/mpi-fixes` @ `4177e22`,
`ocsmesh_mpi_test` @ `5f04d85`. Jobs 9614168–9614171.

## What is new vs the previous comment

- **A and B**: reproduced the same results (mpi 596 s / 1,013 s). No change.
- **C**: reproduced (mpi 4,469 s ≈ prior 4,478 s). `parallel` and `serial_mp`
  **still DNF even with a dedicated 8h job** — the constraint stage alone does
  not fit in 8h for those two modes.
- **D**: **MPI completed** (it did not in the combined run). This is the new
  result and it is significant — see below.

---

## Config D — full recipe minus topofunc (NEW: MPI completed)

Flags: `--skip-topofunc` (adds global `add_contour` + `add_channel` on top of C)

| Mode | Total | Nodes | hfun range |
|------|-------|-------|-----------|
| mpi | **27,111 s (7.5 h)** | **24,972,342** | [1000, 1500] |
| parallel | DNF (8h) | — | — |
| serial_mp | not reached | — | — |

### Finding 1 — `_apply_features` is 71% of runtime and entirely serial on rank 0

cProfile breakdown (Config D / mpi, 27,014 s meshdata):

| Stage | cumtime | % of total |
|-------|---------|-----------|
| `_apply_features` (serial, rank 0) | 19,196 s | **71%** |
| ├─ `_apply_contours` | 8,025 s | 30% |
| ├─ `_apply_channels` | 6,689 s | 25% |
| ├─ constraints + patch/feature | ~4,480 s | 17% |
| `_calculate_and_write_hfun_to_disk_mpi` | 7,099 s | 26% |

`add_contour` + `add_channel` together are **~14,700 s (~4 h)** running
serially on rank 0. Config D now **quantifies** the `TODO(mpi)` in
`collector.py`: parallelizing `_apply_features` is the single largest remaining
speedup opportunity — it is the majority of wall time in a full recipe.

### Finding 2 — the produced mesh is ~25 M nodes (possible recipe over-refinement)

Config D produced **25 million nodes** (vs ~512 K in A–C) and the hfun max
collapsed from 7000 m to **1500 m** — i.e. the global `add_contour`
(`target_size=1500`) + `add_channel` drove almost the whole domain to
near-minimum element size. This 50× node inflation is what pushed the run to
7.5 h.

I believe this is a **recipe-tuning issue on the benchmark side**
(contour/channel `target_size` / `expansion_rate` too aggressive), not an
OCSMesh bug — but it is worth flagging because it dominates the full-recipe
cost and any realistic timing needs a sensible recipe. I will review the
contour/channel parameters before the next run.

---

## Updated status of the "known limitations" list

- **`_apply_features` serial bottleneck** — now quantified at **71%** of a
  full-recipe MPI run (contour 30% + channel 25% + constraints/box 17%).
  Confirmed as the top parallelization target.
- **GEBCO 114M-point sizing field** — still the meshdata outlier in
  parallel/serial_mp; `MAX_SIZING_PTS` cap still recommended.
- **C/D parallel + serial_mp** — still DNF; these modes need the windfall
  partition (24h). For D, note that MPI alone needs 7.5 h, so parallel/serial_mp
  will need well beyond 8h.

## Next

- Review the contour/channel recipe parameters (the 25M-node explosion).
- Re-run C and D `parallel` / `serial_mp` on the windfall partition (24h) to
  capture the mode comparison those two configs are missing.
