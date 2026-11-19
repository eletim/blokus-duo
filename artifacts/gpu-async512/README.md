# Asynchronous fixed-teacher generation
Research only: main, public Web and Frozen gen3 are unchanged. No new training.

The scheduler keeps up to512 independent games alive in eight CPU processes, with per-game generators, RNG, TT and alpha-beta state. Workers write fixed shared-memory slots and send request IDs, not observation arrays. The GPU coordinator drains whichever workers are ready; it does not wait for every cohort. Completed games are replaced by new tasks from the same frozen seed. Each slot has at most one outstanding request, so queue capacity is bounded by concurrency. Descriptors flush after8 requests or a soft1ms check, and after each reply group; no deliberate GPU waiting window is added. Graph capture or CPU work can exceed that soft interval: it is not a hard latency guarantee.

Generation retains width4, completed2/4/6 stages, original candidate ranking, alpha-beta bounds and TT meaning. There is no speculative sibling expansion. CPU8 reference remains the original DeepSearch/native implementation. The optimized input packing and State copy are opt-in to spawned generation workers; normal CPU inference is retained.

CUDA graph buckets use float32 and fixed batch/candidate shapes, including padding. They do not train or modify models. CPU/GPU rounding and batch-shape ties can change a small number of decisions; exact CPU singleton equality and numeric tolerances are tested separately. GPU utilization is a diagnostic, not the speed criterion.

Run from repository root with the documented project environment and optional native backend built:

```bash
export PYTHONPATH=.:scripts OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m blokus.ppo_native build
gzip -dk artifacts/gpu-async512/final/bank.json.gz  # once in a fresh checkout
python scripts/async_generation.py --bank artifacts/gpu-async512/final/bank.json \
  --out /path/to/new-generation --concurrency 512 --batch 128 --fast-encoding --graphs
```

`run.json` fixes configuration and model hashes. On interruption, completed atomic gzip game files are retained byte-for-byte; only unfinished games restart from their stored per-game RNG/prefix. This is game-boundary recovery, not serialization of a live DFS stack. A different batch shape after restart may resolve float32 ties differently. Completed timing outputs cannot be silently overwritten; use a new directory for a new measurement. No files from this experiment are public human-game logs.

`monitor_generation.py` wraps a command and records outer wall, GPU utilization/memory and process CPU counters. Per-run timing records also separate packing, graph capture, device H2D, forward, return, synchronization/dispatch overhead, actual batch histograms, occupancy and queue wait. Summed process RSS counts shared pages more than once and is not physical memory consumption.

The earlier24-game CPU12.5s and3072-game approximately25min are roughly1.9 and2.05 games/s, not a250x discrepancy. They have different banks and timing scopes; the earlier training generator used500ms deadlines, while this throughput comparison requires complete6ply. Only the new identical-bank comparisons support speed ratios.

Timing units matter: eager/barrier `pack_seconds` includes host packing and synchronized transfer wall; its `forward_seconds` is host wall through CUDA completion. Graph packing is host packing only, while `h2d_seconds` and `forward_seconds` use CUDA stream events. `sync_dispatch_seconds` is the remaining host dispatch/wait interval and `return_seconds` includes D2H/result handling. Event spans can include stream scheduling gaps; they are not pure summed kernel durations. These component columns must not be directly treated as equivalent kernel-time measurements. The standalone profiler is collected after throughput timing. Worker CPU/wait totals overlap across workers and cannot be added to wall time. Device-wide utilization includes the existing desktop/context.

For exact full-series reproduction, use measured source commit61e6eb3 in a separate checkout, build the optional native backend, decompress `final/bank.json.gz`, and run `scripts/run_async_final.py`. That commit contains the precommitted bank/plan but no completed formal timing outputs. The latest branch retains all results; use a fresh output directory when calling the generation CLI. A resumed run's elapsed time covers only remaining work: do not divide total bank size by that partial runtime to claim throughput.

On this machine the interpreter is `/home/eletim/blokus-duo-ai/.venv/bin/python`. Generic `python` examples assume that environment is activated. Archived `*-games.tar.gz` files contain the original per-game gzip bytes; create the corresponding directory and extract the archive there for replay. Full old training history/index files are referenced by SHA, not copied into the generation/export path.

Three quantities are distinct: active game slots, valid requests in an NN call, and padded CUDA graph batch rows. The coordinator dispatch cap is128; policy/value requests are split, and graph batch rows round up to powers of two with a minimum of8. Candidate padding is recorded separately as `padding_slots/padding_real`. Graphs can save host dispatch time while doing additional padded GPU arithmetic; no claim of lower kernel FLOPs follows from faster end-to-end generation.

Postprocessing uses the preregistered per-NN tolerance on matched inputs. Completed roots use full windows and give values for their own policy-restricted trees. Raw array differences are diagnostic only: float ties can change root ordering or the inner candidate set. Unmatched array entries are not treated as paired values for an identical fixed tree; TT intermediate bounds are never promoted to exact results. The quality gate is legal completed trajectories, disjoint states, frozen hashes, CPU exact reference checks, matched-input NN tolerance and at least99% same-state action agreement.

This accelerated generator specifically requires CUDA plus the native observation backend and Frozen gen3 four-channel models. It is not a replacement for the public CPU-only CLI; `python -m blokus.ppo_play --generation gen3 --device cpu --backend python` remains available. Installation uses the repository environment/locked dependencies; measured hardware was RTX5090 and Ryzen9950X3D, Python3.12.14, torch2.7.1+cu128, NumPy2.5.4. Do not assume the measured ratio on a different CPU/GPU or candidate distribution.

The development concurrency curve has512 total games at each setting and therefore includes fill/tail effects (the512 setting cannot replenish from a second512). It is not a steady-state concurrency scaling law. The formal3072-game bank has1024 tasks per seed and demonstrably replenishes512 live slots. Reported full512-window rates are descriptive finite-run intervals, not a claim that stage distribution is perfectly stationary. Outer wall remains the primary predeclared comparison.

Each formal run is a fresh process, but OS filesystem/driver caches are not forcibly flushed and unrelated desktop services remain running. Thus initialization-inclusive wall is not a cold-machine benchmark. No torch.compile was used; CUDA graph capture and its three warmup forwards per shape are included and separately counted.

Shared slots reserve up to4096 legal candidates each. Exceeding this capacity raises an error rather than truncating candidates or silently changing the teacher. The measured maximum is recorded per group; this fixed capacity and fixed four-channel gen3 model are explicit constraints of the generation-only path.
