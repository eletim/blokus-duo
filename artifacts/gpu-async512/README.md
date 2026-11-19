# Asynchronous fixed-teacher generation
Research only: main, public Web and Frozen gen3 are unchanged. No new training.

The scheduler keeps up to512 independent games alive in eight CPU processes, with per-game generators, RNG, TT and alpha-beta state. Workers write fixed shared-memory slots and send request IDs, not observation arrays. The GPU coordinator drains whichever workers are ready; it does not wait for every cohort. Completed games are replaced by new tasks from the same frozen seed. Each slot has at most one outstanding request, so queue capacity is bounded by concurrency. Descriptors flush after8 requests or a soft1ms check, and after each reply group; no deliberate GPU waiting window is added. Graph capture or CPU work can exceed that soft interval: it is not a hard latency guarantee.

Generation retains width4, completed2/4/6 stages, original candidate ranking, alpha-beta bounds and TT meaning. There is no speculative sibling expansion. CPU8 reference remains the original DeepSearch/native implementation. The optimized input packing and State copy are opt-in to spawned generation workers; normal CPU inference is retained.

CUDA graph buckets use float32 and fixed batch/candidate shapes, including padding. They do not train or modify models. CPU/GPU rounding and batch-shape ties can change a small number of decisions; exact CPU singleton equality and numeric tolerances are tested separately. GPU utilization is a diagnostic, not the speed criterion.

Run from repository root with the documented project environment and optional native backend built:

```bash
export PYTHONPATH=.:scripts OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m blokus.ppo_native build
python scripts/async_generation.py --bank artifacts/gpu-async512/final/bank.json \
  --out /path/to/new-generation --concurrency 512 --batch 128 --fast-encoding --graphs
```

`run.json` fixes configuration and model hashes. On interruption, completed atomic gzip game files are retained byte-for-byte; only unfinished games restart from their stored per-game RNG/prefix. This is game-boundary recovery, not serialization of a live DFS stack. A different batch shape after restart may resolve float32 ties differently. Completed timing outputs cannot be silently overwritten; use a new directory for a new measurement. No files from this experiment are public human-game logs.

`monitor_generation.py` wraps a command and records outer wall, GPU utilization/memory and process CPU counters. Per-run timing records also separate packing, graph capture, device H2D, forward, return, synchronization/dispatch overhead, actual batch histograms, occupancy and queue wait. Summed process RSS counts shared pages more than once and is not physical memory consumption.

The earlier24-game CPU12.5s and3072-game approximately25min are roughly1.9 and2.05 games/s, not a250x discrepancy. They have different banks and timing scopes; the earlier training generator used500ms deadlines, while this throughput comparison requires complete6ply. Only the new identical-bank comparisons support speed ratios.
