# MPI performance fixes and bug fixes for production use on HPC

## Background: what is being tested and how

This PR includes fixes discovered and validated through a structured benchmarking
campaign on the NOAA RDHPC (Hercules) using a dedicated test repository:
[felicio93/ocsmesh_mpi_test](https://github.com/felicio93/ocsmesh_mpi_test).

### The benchmark (`ocsmesh_mpi_test`)

The benchmark builds a realistic STOFS-3D-Atlantic mesh size function using real
CUDEM 1/9" DEMs and GEBCO background (~15 tiles), then runs `hfun.meshdata()`
in three modes back-to-back on a single Hercules node (80 cores, 512 GB RAM,
exclusive allocation):

- **`mpi`** — 1 manager + 79 MPI workers via `MPIExecutor` (the implementation under test)
- **`parallel`** — 79 `multiprocessing.Pool` workers on a single process
- **`serial_mp`** — OCSMesh serial mode, Pool-based feature steps only

Each mode runs the **identical manifest + recipe**, so output meshes are
equivalence-checked. Per-mode wall times, per-stage wall times, and `cProfile`
`.prof` files are saved alongside `.2dm` size-function outputs.

### Smoke-test matrix

To isolate the cost of individual pipeline stages, the campaign uses a
**4-config cost ladder** where each config adds exactly one cost class:

| Config | fast refs | constraints (no topofunc) | contour/channel + boxes |
|--------|-----------|--------------------------|------------------------|
| A      | 1/tile    | —                        | —                      |
| B      | 2/tile    | —                        | —                      |
| C      | 2/tile    | ✓                        | —                      |
| D      | 2/tile    | ✓                        | ✓                      |

`topo_func_constraint` is excluded from all configs because its unpicklable
lambda forces `_apply_constraints` to fall back to serial even in parallel/mpi
mode, which would defeat the mode comparison.

### Key measurements (15 CUDEM tiles + 1 GEBCO, single node, 80 cores)

**Config A** (fast refs only, no constraints):

| Mode      | Total (s) | Speedup vs serial_mp |
|-----------|-----------|----------------------|
| mpi       | 597       | **4.2×**             |
| parallel  | 2,804     | 0.9×                 |
| serial_mp | 2,478     | 1.0× (baseline)      |

**Config B** (2 fast refs on every tile):

| Mode      | Total (s) | Speedup vs serial_mp |
|-----------|-----------|----------------------|
| mpi       | 1,031     | **2.7×**             |
| parallel  | 4,606     | 0.6×                 |
| serial_mp | 2,735     | 1.0× (baseline)      |

**Config C** (+ constraints, no topofunc) — MPI only completed within 8h walltime:

| Mode      | Total (s) | Notes                                       |
|-----------|-----------|---------------------------------------------|
| mpi       | 4,478     | constraint stage = 2,976s via parallel Pool |
| parallel  | DNF       | killed at 8h walltime during constraints    |
| serial_mp | DNF       | not reached                                 |

---

## Changes

### Bug fixes — correctness issues that prevented MPI from running at all

#### 1. `ocsmesh/hfun/collector.py` — Fix `clip()` call for str/Path DEM inputs (`a4b3a1e`)

**Problem:** When a DEM input is passed as a `str` or `Path`,
`HfunCollector.__init__` reassigns `in_item = str(in_item)` and creates a
separate `raster = Raster(in_item)`. The `base_shape` clip branch then called
`in_item.clip(clip_shape)` on the plain string instead of `raster.clip()`,
causing:

```
AttributeError: 'str' object has no attribute 'clip'
```

This hit on every `str`/`Path` + `base_shape` call — i.e. every real-world
benchmark run.

**Fix:** Call `raster.clip(clip_shape)` consistently, matching the `base_mesh`
branch just below it.

---

#### 2. `ocsmesh/mpi.py` — Force spawn start method to prevent Pool worker MPI aborts (`af67f2e`)

**Problem:** Under a SLURM allocation, `multiprocessing` defaults to `'fork'`.
Forked Pool workers inherit the MPI environment variables, causing every worker
to attempt `MPI_Init` — but workers have no PMI server, resulting in:

```
Abort(1090831): PMI2_Job_GetId returned 14
```

flooding stderr for every worker process. This killed all parallel/MPI runs
under SLURM.

**Fix:** Use `set_start_method('spawn', force=True)` in
`_configure_mpi_environment()` to override any previously set start method and
ensure workers start fresh without inheriting the MPI environment.

---

#### 3. `ocsmesh/mpi.py` — Guard `_get_mpi_comm()` against Pool worker processes (`8159367`)

**Problem:** Even with the spawn start method, Pool worker processes re-import
`ocsmesh`, which calls `_configure_mpi_environment()` at import time. Under a
SLURM allocation, this triggered `_get_mpi_comm()` in every worker, attempting
`MPI_Init` in subprocesses that are not MPI ranks — causing aborts.

**Fix:** `_get_mpi_comm()` now returns `None` for non-`'MainProcess'` processes
by checking `multiprocessing.current_process().name`, so MPI is never initialized
in a Pool worker.

---

#### 4. `ocsmesh/hfun/raster.py` — Skip empty feature windows in `add_feature` (`7640b6e`)

**Problem:** `add_feature` built a KDTree from a `points` list that could be
empty when `add_channel` found no channels on a given tile window, causing:

```
ValueError: data must be of shape (n, m), where there are n points of dimension m
  File ".../ocsmesh/hfun/raster.py" in add_feature
    tree = cKDTree(np.array(points))
```

This crashed the entire run mid-way through feature application.

**Fix:** Guard the KDTree construction with `if len(points) == 0: continue` to
skip windows where no feature points exist.

---

### Performance improvements — measured on Hercules

#### 5. `ocsmesh/hfun/collector.py` — MPI now routes through parallel refinement path (`8d98df1`)

**Problem:** The three refinement dispatchers (`_apply_constraints`,
`_apply_flow_limiters`, `_apply_const_val`) used `execution_mode == 'parallel'`
as their condition for the Pool-based path. Under MPI, `execution_mode == 'mpi'`,
so all three fell through to the **serial** branch on rank 0 even though a
Pool-parallel implementation existed and was safe to use there.

Measured impact (7 tiles, `--skip-constraints` active, job 9600559):

- `_apply_constraints_serial` on rank 0: **1,718 s (29 min)**
- Same workload via `_apply_constraints_parallel` in parallel mode: **~14 min**

This was the dominant bottleneck in MPI mode after constraints were skipped, and
it was completely unnecessary.

**Fix:** Add `'mpi'` alongside `'parallel'` in the dispatch condition for all
three methods:

```python
# Before
if self.execution_mode == 'parallel' and self._nprocs > 1:

# After
if self.execution_mode in ('parallel', 'mpi') and self._nprocs > 1:
```

**Safety:** `_apply_features` is already guarded by `if is_manager:` upstream
(line 1077), so only rank 0 ever reaches these dispatchers — spawning a Pool
there is safe. `TopoFuncConstraint` still auto-falls back to serial via the
existing lambda-pickle guard.

**Measured improvement (Config A, 15 tiles):**

| Stage | Before | After |
|-------|--------|-------|
| `_apply_flow_limiters` | 175 s (serial) | 78 s (parallel Pool) |
| `_apply_const_val` | 163 s (serial) | 55 s (parallel Pool) |

---

#### 6. `ocsmesh/hfun/collector.py` — Cap Pool workers at `min(nprocs, n_tasks)` (`4177e22`)

**Problem:** Every `Pool(processes=self._nprocs)` call spawned the full `nprocs`
count (e.g. 79 workers) regardless of how many tasks were in the batch. With 3
flow-limiter tasks and 79 workers, 76 processes spawned, initialized their full
Python/OCSMesh environment, and sat idle until the 3 real workers finished.

This was a primary reason `parallel` mode was slower than `serial_mp` in earlier
runs: Pool spawn/teardown overhead for 79 workers exceeded the parallelization
benefit for small task counts, and 79 simultaneous gmsh processes competed for
memory bandwidth on the same node.

**Fix:** At every `Pool.map()` call site, cap the worker count:

```python
# Before
with Pool(processes=self._nprocs) as p:
    results = p.map(worker, tasks)

# After
n_workers = min(self._nprocs, len(tasks))
with Pool(processes=n_workers) as p:
    results = p.map(worker, tasks)
```

Sites changed (4 total):

- `_apply_constraints_parallel`
- `_apply_flow_limiters_parallel`
- `_apply_const_val_parallel`
- `_calculate_and_write_hfun_to_disk_parallel`

**Measured improvement (Config A, 3 flow-limiter tasks):**

| Stage | Before (79 workers) | After (3 workers) |
|-------|---------------------|-------------------|
| `_apply_flow_limiters_parallel` | 339 s | 96 s |
| `_apply_const_val_parallel` | 324 s | 87 s |

`parallel` mode is now comparable to `serial_mp` for small task counts instead
of being slower.

---

#### 7. `ocsmesh/hfun/raster.py` — Auto-compute stride for gmsh sizing field (`9549d50`)

**Problem:** `meshdata()` passes the full raster grid to gmsh as a background
sizing field via `gmsh.view.addListData()`. For 1/9" CUDEM tiles (~3 m/px), a
single tile window contains ~65 million points. With many workers running
simultaneously in a `Pool`, this stalled or silently OOM'd workers — observed
as a job hanging for >1 hour with no progress after 14 of 15 tiles completed.

**Fix:** When `stride` is not provided by the caller and `hmin` is set,
auto-compute a stride from the raster resolution:

```python
if stride is None and self.hmin is not None:
    dem_res_m = abs(self.dx) * 111_000  # degrees -> metres, mid-lat approx
    if dem_res_m > 0:
        stride = max(1, int(self.hmin / dem_res_m / 2))
```

The formula keeps ~2 sample points per `hmin` interval, which is sufficient for
gmsh to interpolate the background field. The approximation
`1 deg ≈ 111,000 m` is accurate to ~0.5% at mid-latitudes (STOFS domain).
Callers can still pass an explicit `stride=` to override.

**Effect on point counts (hmin = 1000 m):**

| DEM | Resolution | stride | Points/tile |
|-----|-----------|--------|-------------|
| CUDEM 1/9" | ~3 m/px | 166 | ~2,400 (was 65 M — **27,000× reduction**) |
| GEBCO 15" | ~460 m/px | 1 | no change (already small) |

---

#### 8. `ocsmesh/engines/gmsh.py` — Replace `.tolist()` with numpy buffer in `_apply_sizing` (`9549d50`)

**Problem:** `_apply_sizing` built a Python list via `data_block.ravel().tolist()`
before calling `gmsh.view.addListData()`. For large sizing fields, `.tolist()`
converts a numpy array into a Python list of floats — for 65 M points this
allocates ~2 GB of Python objects per worker, independent of the stride issue.

**Fix:** Pass a 1-D numpy array directly. gmsh's Python API accepts any sequence;
a contiguous numpy array avoids the Python-object allocation entirely:

```python
# Before
flat_data = data_block.ravel().tolist()

# After
data_block = np.ascontiguousarray(np.hstack((coords_f64, z_col, val_col)))
flat_data = data_block.ravel()  # 1-D numpy array, no Python list copy
```

---

#### 9. `ocsmesh/hfun/raster.py` — Default gmsh boundary representation to `'adapt'` (`8d98df1`)

**Problem:** The default `bnd_representation='fixed'` in `GmshOptions` locks
the original dense boundary vertices of each tile as hard points. When the
boundary resolution is much finer than the requested element size (e.g. a 3 m
DEM boundary on a 1000 m hfun), this forces many unnecessarily small elements
near the tile boundary, degrading the background mesh quality.

**Fix:** For the gmsh engine, default to `'adapt'`, which resamples boundary
vertices to match the hfun resolution before meshing via
`utils.resample_geom_by_hfun()`. Uses `setdefault` so callers can override, and
is gated on `mesh_engine == 'gmsh'` since `TriangleOptions` does not accept this
kwarg:

```python
if mesh_engine == 'gmsh':
    mesh_options.setdefault('bnd_representation', 'adapt')
```

---

## Commit summary

| Commit    | File(s)                                                    | Change                                                                           |
|-----------|------------------------------------------------------------|----------------------------------------------------------------------------------|
| `a4b3a1e` | `hfun/collector.py`                                        | fix: `raster.clip()` instead of `in_item.clip()` for str/Path inputs            |
| `af67f2e` | `mpi.py`                                                   | fix: use `force=True` in `set_start_method('spawn')` to override fork default   |
| `8159367` | `mpi.py`                                                   | fix: guard `_get_mpi_comm()` against Pool worker processes                       |
| `7640b6e` | `hfun/raster.py`                                           | fix: skip empty feature windows in `add_feature` to avoid cKDTree crash          |
| `8d98df1` | `hfun/collector.py`, `hfun/raster.py`, `engines/gmsh.py`  | feat: route mpi through parallel refinement path; default gmsh boundary to adapt |
| `9549d50` | `hfun/raster.py`, `engines/gmsh.py`                        | fix: auto-compute stride; replace `.tolist()` with numpy buffer in gmsh          |
| `4177e22` | `hfun/collector.py`                                        | perf: cap Pool workers at `min(nprocs, n_tasks)` to avoid idle-worker overhead   |

> Note: commits `4174611` and `6ba3769` (add/remove diagnostic prints in
> `_configure_mpi_environment`) are also on the branch but contain no
> functional changes and can be squashed before merge.

---

## Testing

All changes were validated on Hercules (NOAA RDHPC) using the smoke-test matrix
described above. Full benchmark results, per-mode `cProfile` `.prof` files, and
`.2dm` hfun outputs are available in the `ocsmesh_mpi_test` repository under
`results/smoke_matrix_9611495/`.

Equivalence checks confirmed that all completed modes (Config A: all three
modes; Config B: all three modes; Config C: mpi only) produced matching
node/triangle counts and identical hfun value ranges (`hfun=[1000, 7000]`),
verifying correctness across modes.

---

## Known limitations / follow-up work

- **GEBCO tile sizing-field cap:** stride=1 on GEBCO still yields ~114 M points
  because the tile is geographically large (global extent), not fine-resolution.
  A `MAX_SIZING_PTS` hard cap in `meshdata()` would fix this without affecting
  CUDEM tiles.
- **Config C/D parallel and serial_mp** not measured due to 8h walltime. The
  quantified benefit of constraint parallelization vs serial for those modes is
  still outstanding.
- **`_apply_features` is still serial on rank 0:** constraints, contour/channel,
  and box refinements all run on rank 0 only (see `TODO(mpi)` in `collector.py`
  line 1074). This is the next major parallelization target.
- **`TopoFuncConstraint` forces serial fallback:** its stored lambda cannot be
  pickled for Pool. This should be addressed so constraint parallelization is
  not disabled by default in realistic recipes that include function-based
  constraints.
