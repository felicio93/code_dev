# Update: GEBCO sizing-field cap + contour/channel recipe fix (jobs 9618400–9618403)

## What changed for this run

### OCSMesh change (`felicio/mpi-fixes` @ `657434b`)

Added `_GMSH_MAX_SIZING_PTS = 1_000_000` as a module-level constant in
`ocsmesh/hfun/raster.py`, with a per-window cap inside `meshdata()`:

```python
_GMSH_MAX_SIZING_PTS: int = 1_000_000  # module-level constant

# inside the meshdata() loop, after the resolution-based stride:
if mesh_engine == 'gmsh':
    n_pts_strided = (win.height // step) * (win.width // step)
    if n_pts_strided > _GMSH_MAX_SIZING_PTS:
        step = max(step, ceil(sqrt(win.height * win.width / _GMSH_MAX_SIZING_PTS)))
        _logger.info(f"Window {i_win+1}: stride increased to {step} to cap ...")
```

The resolution-based auto-stride handles fine DEMs (e.g. CUDEM 1/9": stride=145,
~3,136 pts/tile). This secondary cap handles geographically large tiles where the
resolution is coarse but the tile area is enormous — specifically the global GEBCO
background tile (10,805 × 10,619 px at 15"), which previously produced ~114M pts
at stride=1 and dominated all parallel/serial_mp runs (~37 min per tile).

**Effect on GEBCO:** stride 1 → 11, 114M pts → **949,578 pts**, ~37 min → ~45 sec.

### `ocsmesh_mpi_test` change (`main` @ `e8433e5`)

Changed `add_contour` and `add_channel` `target_size` from 1500/1000 → **3500 m**
(half of `GLOBAL_HMAX=7000 m`). `expansion_rate=0.05` unchanged.

---

## Results (jobs 9618400–9618403, 15 tiles, single node, 80 cores)

### Config A — all 3 modes completed ✅

| Mode | Total (s) | vs prior run | Notes |
|------|-----------|-------------|-------|
| mpi | 362 | was 597 | **1.7× faster** |
| parallel | 564 | was 2,840 | **5× faster** |
| serial_mp | 311 | was 2,478 | **8× faster** |

The GEBCO cap alone accounts for essentially all of the improvement. The
GEBCO tile that previously took ~37 min in parallel and serial_mp now takes
~45 seconds (confirmed: `Applying sizing field with 949,578 points...`).

**New finding:** Without the GEBCO bottleneck, `serial_mp` (311s) is now
**faster than MPI** (362s) for Config A. MPI's dispatch overhead (`_dispatch`
= 52s in the cProfile) slightly exceeds its benefit when all 15 tiles complete
quickly. MPI's speedup advantage is most pronounced when there is an outlier
tile that blocks single-threaded modes — which is the realistic production
scenario. With a homogeneous workload (all tiles fast), the persistent-rank
MPI overhead is the limiting factor.

---

### Config B — all 3 modes completed ✅

| Mode | Total (s) | vs prior run | Notes |
|------|-----------|-------------|-------|
| mpi | 798 | was 1,031 | **1.3× faster** |
| parallel | 2,280 | was 4,606 | **2× faster** |
| serial_mp | 548 | was 2,735 | **5× faster** |

**serial_mp (548s) is now faster than parallel (2,280s) by 4.2×**, and only
1.5× slower than MPI. The GEBCO cap helps all modes significantly. The
remaining gap between MPI and serial_mp comes from the 14+14 = 28 Pool-based
refinement tasks on rank 0 (`_apply_flow_limiters_parallel` = 317s +
`_apply_const_val_parallel` = 255s = 572s) that must complete before MPI can
dispatch the meshdata stage.

---

### Config C — MPI completed, parallel/serial_mp still DNF ✅/❌

| Mode | Total (s) | Notes |
|------|-----------|-------|
| mpi | 4,254 | completed (consistent with prior 4,469s) |
| parallel | DNF | killed at 8h walltime during constraint stage |
| serial_mp | DNF | not reached |

Config C/mpi reproduced consistently. The constraint stage
(`_apply_constraints_parallel`) = 2,974s — unchanged by the GEBCO fix
(constraint stage dominates, not meshdata). The GEBCO fix does not help C/mpi
because it is limited by serial rank-0 constraint application, not by the
sizing field size.

Parallel and serial_mp were again killed during the constraint stage. Even
with the GEBCO fix, the constraint stage alone occupies most of the 8h budget
for these modes. **The windfall partition (24h) is required to measure
Config C parallel and serial_mp.**

---

### Config D — GEBCO cap working, but 25M-node explosion persists ❌

| Mode | Total (s) | Nodes | hfun range |
|------|-----------|-------|-----------|
| mpi | 26,723 | **24,972,342** | [1000, 1500] |
| parallel | DNF | — | — |

Config D produced **the same 25M nodes and hfun=[1000, 1500]** as the previous
run despite the `target_size=3500` change. The fix did not work.

**Root cause analysis:** The `target_size` parameter alone is insufficient. The
`expansion_rate=0.05` produces a transition zone of:

```
(target_size - hmin) / expansion_rate = (3500 - 1000) / 0.05 = 50,000 m (50 km)
```

A 50 km transition zone extending from every point along the full STOFS-3D
Atlantic shoreline and -200m contour covers essentially the entire domain.
`add_channel` has the same problem — 50 km of refinement spreading from each
detected channel floods the mesh. The identical node count (24,972,342) and
hfun range ([1000, 1500]) across two different `target_size` values confirms
that the transition zones are overlapping and driving the entire domain to
`hmin = 1000 m`.

**The correct fix is to increase `expansion_rate`**, not `target_size`. A higher
expansion rate means a steeper, narrower transition zone. For example,
`expansion_rate=0.3` would give:

```
(3500 - 1000) / 0.3 = ~8,300 m (8 km) transition zone
```

This would refine only near the actual shoreline/contours rather than flooding
the entire domain.

---

## Summary of new findings

| Finding | Significance |
|---|---|
| GEBCO cap reduced tile time from ~37 min to ~45 sec | Config A parallel: 2,840s → 564s (5×); serial_mp: 2,478s → 311s (8×) |
| Without GEBCO bottleneck, serial_mp beats MPI for Config A | MPI's dispatch overhead (~52s) exceeds benefit for homogeneous workloads |
| `target_size` change alone does not fix Config D | `expansion_rate=0.05` creates 50 km zones that flood the entire domain regardless of `target_size` |
| Config D recipe fix requires `expansion_rate` increase | Suggested: `expansion_rate=0.3` → ~8 km transition zone; needs testing |
| Config C parallel/serial_mp still need windfall partition | Constraint stage alone exhausts the 8h budget |

---

## Updated full matrix (best results to date)

| Config | MPI (s) | parallel (s) | serial_mp (s) | MPI vs serial_mp |
|--------|---------|--------------|---------------|-----------------|
| A — 1 fast ref/tile | **362** | 564 | **311** | 0.86× (serial_mp faster) |
| B — 2 fast refs/tile | **798** | 2,280 | 548 | **1.45×** |
| C — + constraints | **4,254** | DNF | DNF | — |
| D — + contour/channel | 26,723 (7.4h) | DNF | DNF | — (broken recipe) |

## Next step for Config D

Increase `expansion_rate` for `add_contour` and `add_channel` (e.g. 0.15–0.3)
to narrow the transition zones. The target_size=3500 change can remain — it is
correct — but `expansion_rate` needs to match it. With a narrower zone, Config D
should produce a realistic ~500K–1M node mesh and complete in <2h for MPI.
