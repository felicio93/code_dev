"""
Hierarchical Manager-Worker MPI pattern for OCSMesh Phase 2.

Topology (example: 2 nodes, 4 ranks each = 8 ranks total):

  Rank 0  -- Global Coordinator (1 per job)
  Rank 1  -- Node Coordinator   (1 per node, does not compute)
  Rank 5  -- Node Coordinator   (1 per node, does not compute)
  Rank 2,3,4   -- Workers on node 0
  Rank 6,7,8   -- Workers on node 1 (hypothetical 9-rank example)

  Global Coordinator (rank 0)
      |
      |---> Node Coordinator (rank 1, node 0)
      |         |---> Worker rank 2
      |         |---> Worker rank 3
      |         `---> Worker rank 4
      |
      `---> Node Coordinator (rank 5, node 1)
                |---> Worker rank 6
                |---> Worker rank 7
                `---> Worker rank 8

Message flow:
  Workers -> Node Coordinator: "I am free / here is my result"
  Node Coordinator -> Global Coordinator: "My node has capacity / here are results"
  Global Coordinator -> Node Coordinator: "Here is a batch of tasks"
  Node Coordinator -> Workers: "Here is your task"

When to use this over PR #248's flat manager:
  Rule of thumb: when your job regularly exceeds ~128 workers AND DEM tiles
  are fast enough (< 5 seconds each) that Rank 0's message handling becomes
  measurable. For tiles that take 30+ seconds, PR #248's flat manager will
  never be the bottleneck -- task runtime dominates completely. For OCSMesh
  with large DEM tiles, you likely won't need this until running at 500+ cores
  with very small tiles.
"""

import sys
from typing import Any, List, Dict, Optional

try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False


# ── Message tags ──────────────────────────────────────────────────────────────
TAGS = {
    'TASK':       1,   # coordinator -> worker / global -> node-coord
    'RESULT':     2,   # worker -> node-coord / node-coord -> global
    'ERROR':      3,   # worker -> node-coord / node-coord -> global
    'STOP':       4,   # shutdown signal (propagates down the tree)
    'NODE_READY': 5,   # node-coord -> global: "send me a batch"
    'BATCH':      6,   # global -> node-coord: list of tasks
}


# ── Topology helpers ──────────────────────────────────────────────────────────

def build_topology(comm) -> Dict:
    """
    Assign roles based on MPI rank and node name.

    Returns a dict describing this rank's role and its peers.
    MPI_Get_processor_name() gives the node name, so ranks are
    grouped by node automatically.

    Returns
    -------
    dict with keys:
        role         : 'global_coord' | 'node_coord' | 'worker'
        rank         : int
        size         : int
        node_id      : int  (which node this rank lives on)
        global_coord : int  (rank of the global coordinator, always 0)
        node_coord   : int  (rank of this rank's node coordinator)
        workers      : list[int]  (worker ranks on this node, node_coord only)
        node_coords  : list[int]  (all node coordinators, global_coord only)
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Get node name for each rank -- gather all to rank 0
    my_node = MPI.Get_processor_name()
    all_nodes = comm.allgather(my_node)

    # Build node -> rank list mapping
    node_to_ranks: Dict[str, List[int]] = {}
    for r, node in enumerate(all_nodes):
        node_to_ranks.setdefault(node, []).append(r)

    # Sort nodes deterministically
    sorted_nodes = sorted(node_to_ranks.keys())

    # Rank 0 is always the global coordinator.
    # First rank on each node (excluding rank 0) becomes the node coordinator.
    global_coord = 0
    node_coords = []
    node_workers = {}  # node_coord_rank -> [worker ranks]

    for node_name in sorted_nodes:
        ranks_on_node = sorted(node_to_ranks[node_name])
        candidates = [r for r in ranks_on_node if r != global_coord]
        if not candidates:
            # Edge case: global coordinator is alone on its node.
            continue
        nc = candidates[0]
        node_coords.append(nc)
        node_workers[nc] = candidates[1:]  # remaining ranks are workers

    # Determine this rank's role
    node_id = sorted_nodes.index(all_nodes[rank])

    if rank == global_coord:
        role = 'global_coord'
    elif rank in node_coords:
        role = 'node_coord'
    else:
        role = 'worker'

    # Find this rank's node coordinator
    my_node_coord = None
    for nc, workers in node_workers.items():
        if rank == nc or rank in workers:
            my_node_coord = nc
            break
    if rank == global_coord:
        my_node_coord = global_coord

    return {
        'role':        role,
        'rank':        rank,
        'size':        size,
        'node_id':     node_id,
        'global_coord': global_coord,
        'node_coord':  my_node_coord,
        'workers':     node_workers.get(rank, []),    # non-empty for node_coord
        'node_coords': node_coords,                   # non-empty for global_coord
    }


# ── Role implementations ───────────────────────────────────────────────────────

def run_global_coordinator(comm, topo: Dict, tasks: List[Dict]) -> List[Dict]:
    """
    Rank 0 only.

    Streams batches of tasks to node coordinators on demand.
    Each node coordinator signals when it has free worker capacity;
    the global coordinator sends it one task per free worker slot.

    This is the key scalability improvement over a flat manager:
    instead of tracking N individual workers, rank 0 only talks to
    M node coordinators (M = number of nodes), so message rate
    scales with nodes, not cores.
    """
    task_queue = list(tasks)
    results = []
    node_coords = topo['node_coords']
    active_nodes = set(node_coords)

    # Seed: send initial batch to every node coordinator
    for nc in node_coords:
        n_workers = _query_node_worker_count(comm, nc)
        batch = _pop_batch(task_queue, n_workers)
        if batch:
            comm.send(batch, dest=nc, tag=TAGS['BATCH'])
        else:
            comm.send(None, dest=nc, tag=TAGS['STOP'])
            active_nodes.discard(nc)

    # Refill loop: wait for any node coordinator to report back
    while active_nodes:
        status = MPI.Status()
        message = comm.recv(
            source=MPI.ANY_SOURCE,
            tag=MPI.ANY_TAG,
            status=status
        )
        src = status.Get_source()
        tag = status.Get_tag()

        if tag in (TAGS['RESULT'], TAGS['ERROR']):
            # message is a list of result dicts from that node's workers
            results.extend(message)

            # Refill that node coordinator with more tasks
            n_free = len(message)   # one result per finished worker
            batch = _pop_batch(task_queue, n_free)
            if batch:
                comm.send(batch, dest=src, tag=TAGS['BATCH'])
            else:
                comm.send(None, dest=src, tag=TAGS['STOP'])
                active_nodes.discard(src)

    return results


def run_node_coordinator(comm, topo: Dict):
    """
    One per node (not rank 0).

    Receives task batches from the global coordinator, fans them out
    to local workers via point-to-point send/recv (identical to
    PR #248's current flat manager, but scoped to one node).
    Aggregates local results and forwards them upstream.

    Intra-node communication uses MPI's shared-memory transport,
    making the local fan-out essentially free compared to cross-node
    messages.
    """
    workers = topo['workers']
    global_coord = topo['global_coord']

    if not workers:
        # Degenerate case: node has only one rank (the coordinator itself).
        # Fall back to running tasks directly without delegating.
        _run_as_solo_node(comm, topo, global_coord)
        return

    idle_workers = list(workers)
    inflight: Dict[int, Dict] = {}   # worker_rank -> task

    while True:
        # Wait for a batch from the global coordinator
        status = MPI.Status()
        message = comm.recv(source=global_coord, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()

        if tag == TAGS['STOP']:
            # Drain any in-flight workers first, then shut them down
            _drain_workers(comm, inflight, workers)
            for w in workers:
                comm.send(None, dest=w, tag=TAGS['STOP'])
            break

        batch = message
        local_results = []

        # Distribute batch tasks to idle workers
        for task in batch:
            if idle_workers:
                w = idle_workers.pop(0)
                comm.send(task, dest=w, tag=TAGS['TASK'])
                inflight[w] = task
            else:
                # All workers busy -- wait for one to finish before sending
                result, finished_worker = _wait_for_one_worker(
                    comm, workers, inflight)
                local_results.append(result)
                comm.send(task, dest=finished_worker, tag=TAGS['TASK'])
                inflight[finished_worker] = task

        # Drain remaining in-flight tasks for this batch
        while inflight:
            result, finished_worker = _wait_for_one_worker(
                comm, workers, inflight)
            local_results.append(result)
            idle_workers.append(finished_worker)

        # Report all results for this batch upstream
        has_errors = any(
            r.get('status') == 'error'
            for r in local_results
            if isinstance(r, dict)
        )
        reply_tag = TAGS['ERROR'] if has_errors else TAGS['RESULT']
        comm.send(local_results, dest=global_coord, tag=reply_tag)


def run_worker(comm, topo: Dict, worker_fn):
    """
    All non-coordinator ranks.

    Identical in structure to PR #248's _run_worker().
    Receives one task at a time from its node coordinator,
    executes it, returns the result.

    Intra-node communication uses MPI shared memory -- effectively
    zero-cost compared to cross-node sends.
    """
    node_coord = topo['node_coord']
    rank = topo['rank']

    while True:
        status = MPI.Status()
        message = comm.recv(source=node_coord, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()

        if tag == TAGS['STOP']:
            break

        task = message
        try:
            result = worker_fn(task)
            result['worker_rank'] = rank
            comm.send(result, dest=node_coord, tag=TAGS['RESULT'])
        except Exception as exc:
            comm.send(
                {
                    'status': 'error',
                    'worker_rank': rank,
                    'original_index': task.get('original_index', -1),
                    'error': repr(exc),
                },
                dest=node_coord,
                tag=TAGS['ERROR']
            )


# ── Public entry point ─────────────────────────────────────────────────────────

class HierarchicalMPITaskRunner:
    """
    Drop-in replacement for PR #248's MPITaskRunner for large-scale jobs.

    Usage (identical to #248 from the user's perspective)::

        runner = HierarchicalMPITaskRunner()

        def main():
            hfun = Hfun(raster_list)
            hfun.execution_mode = 'mpi'
            return hfun.meshdata()

        results = runner.run(main)  # all ranks call this -- no rank checks

    Comparison with PR #248 flat manager
    -------------------------------------

    PR #248 (flat):                    Phase 2 (hierarchical):

    Rank 0 <-> Rank 1                  Rank 0 <-> Node Coord A <-> Worker 1
    Rank 0 <-> Rank 2                             (shared mem)  <-> Worker 2
    Rank 0 <-> Rank 3                             (shared mem)  <-> Worker 3
    ...                                Rank 0 <-> Node Coord B <-> Worker 4
    Rank 0 <-> Rank 999                           (shared mem)  <-> Worker 5
                                                  (shared mem)  <-> Worker 6
    Rank 0 handles 999 messages
    at peak throughput                 Rank 0 handles M node messages
                                       where M = number of nodes (e.g. 50)
    """

    def __init__(self):
        if not MPI_AVAILABLE:
            self.comm = None
            self.rank = 0
            self.size = 1
            self.topo = {'role': 'worker'}
            return

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.topo = build_topology(self.comm)

        # Install global excepthook -- same safety net as PR #248
        _install_excepthook(self.comm)

    def run(self, user_fn) -> Optional[List[Dict]]:
        """
        All ranks call this. Only the global coordinator returns results.

        Parameters
        ----------
        user_fn : callable
            On Rank 0: must return a list of task dicts.
            On worker ranks: not called.
        """
        if self.size == 1 or not MPI_AVAILABLE:
            return _run_sequential(user_fn)

        role = self.topo['role']

        if role == 'global_coord':
            tasks = user_fn()
            return run_global_coordinator(self.comm, self.topo, tasks)

        elif role == 'node_coord':
            run_node_coordinator(self.comm, self.topo)
            return None

        else:  # worker
            registry = _worker_registry()
            run_worker(
                self.comm, self.topo,
                worker_fn=lambda task: registry[task['op']](task)
            )
            return None

    def dispatch(self, tasks: List[Dict]) -> List[Dict]:
        """
        Rank-0-only convenience method (mirrors PR #248 MPITaskRunner.dispatch).
        """
        if self.topo['role'] != 'global_coord':
            raise RuntimeError("dispatch() must only be called by Rank 0")
        return run_global_coordinator(self.comm, self.topo, tasks)


# ── Internal helpers ───────────────────────────────────────────────────────────

def _pop_batch(queue: list, n: int) -> list:
    """Pop up to n items from the front of queue in-place."""
    batch = queue[:n]
    del queue[:n]
    return batch


def _query_node_worker_count(comm, node_coord_rank: int) -> int:
    """
    Return the number of workers managed by a node coordinator.

    In a production implementation, store worker counts at topology
    build time (no message needed). This placeholder assumes a uniform
    distribution of 4 workers per node.
    """
    return 4


def _wait_for_one_worker(comm, workers: list, inflight: dict):
    """Block until any worker sends a result; return (result, worker_rank)."""
    MPI = sys.modules.get('mpi4py.MPI') or __import__('mpi4py').MPI
    status = MPI.Status()
    result = comm.recv(
        source=MPI.ANY_SOURCE,
        tag=MPI.ANY_TAG,
        status=status
    )
    finished = status.Get_source()
    inflight.pop(finished, None)
    return result, finished


def _drain_workers(comm, inflight: dict, workers: list):
    """Collect all outstanding results from in-flight workers."""
    while inflight:
        _wait_for_one_worker(comm, workers, inflight)


def _run_as_solo_node(comm, topo: dict, global_coord: int):
    """
    Fallback for a node with only one rank (the coordinator itself).
    Runs tasks sequentially and reports results directly to the global
    coordinator.
    """
    registry = _worker_registry()
    while True:
        status = MPI.Status()
        msg = comm.recv(source=global_coord, tag=MPI.ANY_TAG, status=status)
        if status.Get_tag() == TAGS['STOP']:
            break
        results = []
        for task in msg:
            try:
                result = registry[task['op']](task)
                results.append(result)
            except Exception as exc:
                results.append({
                    'status': 'error',
                    'original_index': task.get('original_index', -1),
                    'error': repr(exc),
                })
        comm.send(results, dest=global_coord, tag=TAGS['RESULT'])


def _run_sequential(user_fn):
    """Single-process fallback (no MPI or size == 1)."""
    tasks = user_fn()
    registry = _worker_registry()
    results = []
    for task in tasks:
        try:
            results.append(registry[task['op']](task))
        except Exception as exc:
            results.append({
                'status': 'error',
                'original_index': task.get('original_index', -1),
                'error': repr(exc),
            })
    return results


def _worker_registry() -> Dict:
    """Map operation name -> worker function. Mirrors PR #248's registry."""
    from ocsmesh.hfun.collector import _meshdata_task_worker
    return {'meshdata': _meshdata_task_worker}


def _install_excepthook(comm):
    """
    Override sys.excepthook to call comm.Abort(1) on any uncaught exception.
    Prevents zombie HPC/cloud processes when one rank crashes.
    Identical in purpose to MPITaskRunner.install_mpi_excepthook() in PR #248.
    """
    import traceback
    rank = comm.Get_rank()

    def _hook(exctype, value, tb):
        sys.stderr.write(f"[Rank {rank}] uncaught exception -- aborting:\n")
        traceback.print_exception(exctype, value, tb, file=sys.stderr)
        sys.stderr.flush()
        comm.Abort(1)

    sys.excepthook = _hook
