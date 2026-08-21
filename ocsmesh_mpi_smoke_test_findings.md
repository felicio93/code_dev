# OCSMesh MPI Smoke Test Findings — Job 9611495

Benchmark run on Hercules (single node, 80 cores, 512 GB RAM, exclusive allocation).
15 CUDEM 1/9" tiles + 1 GEBCO background tile. All three modes run back-to-back
on the same node from `slurm_smoke_matrix.sh`.

---

## Config A — no constraints, 1 fast ref/tile

**Flags:** `--skip-constraints --skip-box-refinements --light-features`

Only `add_subtidal_flow_limiter` (3 tiles) and `add_constant_value` (3 tiles)
run before meshdata dispatch.

### Results

| Mode      | Total (s) | meshdata (s) | Speedup vs serial_mp |
|-----------|-----------|--------------|----------------------|
| mpi       | 597       | 504          | **4.2×**             |
| parallel  | 2,804     | 2,742        | 0.9×                 |
| serial_mp | 2,478     | 2,418        | 1.0× (baseline)      |

### What we learned

**The Pool-worker cap fix worked.** Before capping Pool workers at
`min(nprocs, n_tasks)`, the refinement stages were spawning 79 workers for
only 3 tasks. After the fix:

| Stage | Before (79 workers) | After (3 workers) |
|-------|---------------------|-------------------|
| `_apply_flow_limiters_parallel` | 339 s | 96 s |
| `_apply_const_val_parallel` | 324 s | 87 s |

**MPI is 4.2× faster than serial_mp.** This is the cleanest MPI speedup number
in the matrix because Config A has almost no serial overhead on rank 0 — the
Pool refinements are fast (3 tasks, 3 workers) and the meshdata dispatch is
fully MPI-parallelized across 79 workers.

**Parallel (2,804 s) is now comparable to serial_mp (2,478 s).** Before the
Pool-worker cap fix, parallel was much worse than serial_mp. The modes are now
in the expected order: MPI >> serial_mp ≈ parallel.

**The GEBCO tile dominates parallel and serial_mp.** One of the 15 tiles is
GEBCO (~460 m/px), which gets `stride=1` from the auto-stride formula (correct
— it is coarse-resolution). But GEBCO covers a large geographic extent, so
stride=1 still produces ~114 million points. Passing 114M points to
`gmsh.view.addListData()` takes ~37 minutes per mode. This single tile accounts
for most of the difference between MPI (597 s) and the other two modes:

- In **MPI** mode: GEBCO is dispatched to one of the 79 workers and runs in
  parallel with the other 14 tiles — it does not block anything.
- In **parallel** and **serial_mp** modes: the GEBCO tile sits in the
  Pool/serial queue and holds up the final assembly for ~37 minutes.

This is a known follow-up fix: a `MAX_SIZING_PTS` cap in `meshdata()` would
subsample GEBCO to a manageable size without affecting mesh quality (gmsh
interpolates the background field spatially — it does not need 114M input
points for a 1000 m hfun).

---

## Config B — no constraints, 2 fast refs on every tile

**Flags:** `--all-fast-refinements`

Both `add_subtidal_flow_limiter` AND `add_constant_value` applied to all 14
CUDEM tiles (instead of 3+3 via the modulo scheme).

### Results

| Mode      | Total (s) | meshdata (s) | Speedup vs serial_mp |
|-----------|-----------|--------------|----------------------|
| mpi       | 1,031     | 940          | **2.7×**             |
| parallel  | 4,606     | 4,545        | 0.6×                 |
| serial_mp | 2,735     | 2,671        | 1.0× (baseline)      |

### What we learned

**More refinement tasks hurts parallel more than MPI or serial_mp.**
With 28 refinement tasks (14 flow + 14 const), parallel is now *slower* than
serial_mp despite using 14 workers per Pool call.

The reason is that each refinement task is IO-bound (reading and writing raster
`.tif` files via the memcache). The overhead of:

1. Pickling task arguments and shipping to workers
2. Workers reading raster data from disk
3. Pickling results and returning to manager
4. Manager reloading into `HfunCollector` state (14 updates × 2 stages)

exceeds the parallelization benefit when the compute-per-task is dominated by
IO. Serial mode avoids all IPC and simply reads/writes files sequentially.

**MPI is still 2.7× faster than serial_mp**, but the absolute time is higher
than Config A because 14+14 = 28 refinement tasks on rank 0 via Pool take
longer than 3+3 = 6 tasks, and rank 0 cannot start the MPI meshdata dispatch
until all refinements are done.

**The GEBCO tile dominates parallel and serial_mp again** (~37 min each), for
the same reason as Config A.

---

## Config C — constraints (no topofunc) + box refinements

**Flags:** `--skip-topofunc --light-features`

Adds `topo_bound_constraint` (2 tiles), `courant_num_constraint` (2 tiles),
`add_region_constraint` (BOX1), `add_patch` (BOX2), and `add_feature` (BOX2
line) on top of the fast refinements from Config A.

### Results

| Mode      | Total (s) | Notes                                          |
|-----------|-----------|------------------------------------------------|
| mpi       | 4,478     | completed; constraint stage = 2,976 s          |
| parallel  | DNF       | killed at 8h walltime during constraint stage  |
| serial_mp | DNF       | never reached                                  |

### What we learned

**Constraints are expensive even with the new parallel routing.**
The `_apply_constraints_parallel` stage took **2,976 s (~50 min)** on rank 0
even though it used the Pool-parallel path (the new behavior after the
`execution_mode in ('parallel', 'mpi')` fix). This is the `_apply_rate` / KDTree
distance expansion path that is inherently expensive per tile — running it via
a Pool helps vs serial, but the per-tile cost (~200 s/tile × 15 tiles) dominates.

**The MPI meshdata dispatch itself was only 355 s** — essentially unchanged from
Config A. This confirms that `_apply_constraints` (running on rank 0 before
dispatch) is the bottleneck, not the MPI parallelism.

**Parallel and serial_mp were killed by the 8h walltime** during their
constraint stages — they never completed. We cannot yet quantify how much the
`mpi`→parallel routing fix helped for the constraint stage in those modes.

**The cProfile confirms the constraint path cost breakdown (Config C / mpi):**

| Function | cumtime |
|---|---|
| `_apply_features` | 4,018 s |
| `_apply_constraints_parallel` | 2,976 s |
| `add_feature` (box + patch) | ~900 s |
| `_calculate_and_write_hfun_to_disk_mpi` | 355 s |
| MPI dispatch | 290 s |

The constraint stage (`_apply_constraints`) accounts for **66% of total MPI
wall time** in Config C. The actual MPI-parallelized meshdata stage is only 8%.

---

## Config D — full recipe minus topofunc

**Flags:** `--skip-topofunc`

Adds global `add_contour` and `add_channel` (the O(tiles × segments)
bottleneck) on top of Config C.

### Result

**Never reached** — the job ran out of time during Config C / parallel.

The cost of global contour/channel on rank 0 was not measured. This is the
`TODO(mpi)` in `collector.py` (line 1074) and the next major parallelization
target.

---

## Summary: what we learned overall

| Question | Answer |
|---|---|
| Does MPI actually parallelize meshdata? | Yes — 4.2× speedup on a clean workload (Config A). |
| Did the `mpi` → parallel routing fix help? | Yes — MPI refinements dropped from serial (~175 s) to Pool (~78 s). |
| Did the Pool-worker cap fix help? | Yes — refinement stages 3.5× faster; parallel now comparable to serial_mp. |
| Why is parallel sometimes slower than serial_mp? | IO-bound raster tasks: Pool IPC and result-serialization overhead exceeds parallelization benefit. |
| What dominates Config C MPI time? | `_apply_constraints_parallel` on rank 0: 2,976 s out of 4,478 s total (66%). |
| What is the remaining serial bottleneck? | (1) GEBCO tile (114M pts, stride=1), (2) `_apply_constraints` / `_apply_rate` KDTree per-tile cost, (3) `_apply_features` (contour/channel) not yet MPI-distributed. |
| What is the next OCSMesh target? | (1) `MAX_SIZING_PTS` cap for GEBCO-type tiles in `meshdata()`, (2) parallelizing `_apply_features` on rank 0 (the `TODO(mpi)` in `collector.py`). |

---

## Key numbers at a glance

| Config | MPI (s) | parallel (s) | serial_mp (s) | MPI speedup vs serial_mp |
|--------|---------|--------------|---------------|--------------------------|
| A — 1 fast ref/tile | 597 | 2,804 | 2,478 | **4.2×** |
| B — 2 fast refs/tile | 1,031 | 4,606 | 2,735 | **2.7×** |
| C — + constraints | 4,478 | DNF | DNF | — |
| D — + contour/channel | DNF | DNF | DNF | — |
