# Reproduce GPU generation benchmark

Run from this research checkout. Python3.12, NumPy, PyTorch2.7.1+cu128;
RTX5090/CUDA are needed only for the broker mode. CPU inference remains unchanged.
Use the same pinned native backend as the prior placement study:

```bash
python -m blokus.ppo_native build
export PYTHONPATH=.:scripts OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python scripts/check_gpu_generation.py
python scripts/gpu_generation.py cpu --repeat 101
python scripts/gpu_generation.py gpu --repeat 101
python scripts/gpu_generation.py gpu --repeat 102
python scripts/gpu_generation.py cpu --repeat 102
python scripts/gpu_generation.py rebuild-exclusions
python scripts/audit_gpu_generation.py  # audit the retained original repeats 1/2
python -m pytest -q tests/test_batched_inference.py tests/test_ppo_deeper.py
```

The saved bank is immutable. `prepare` requires the original training state index
(`/tmp/ppo-lr64-training.npy`) and historical artifact references; do not regenerate
it to reproduce this test. Inference/export itself never requires that index.
Audit also requires the index. `excluded.json` is a locally rebuilt union of prior
state hashes and is not duplicated in Git; `exclusion-manifest.json` records its
hash and inputs. Existing game files are verified against the bank and skipped.
A partially played game has no final JSON; restarting uses its saved initial RNG
and replays it from the prefix. Completion files are atomically renamed. This is
game-boundary resume, not live node/worker checkpointing. Do not interpret a resumed
run's elapsed time as full-bank throughput; timing reports count only new games.
For fresh timing, choose unused repeat numbers (do not erase prior results).

Requests carry worker/model/serial identifiers. The broker never advances games
or generates random choices. CPU workers retain legal rules, policy tie handling,
alpha-beta, TT bound semantics and completed-iteration fallback. GPU batches are
grouped by frozen seed and policy/value input, capped32 with1ms collection window.
Microbatch composition depends on scheduling; float32 arithmetic may differ from
CPU and between batch sizes. Full bitwise trajectory reproducibility is conditional
on the measured ranking/move margins, not promised for arbitrary positions.

CPU-only use of the frozen exports is unchanged:

```bash
python -m blokus.ppo_play --help
python scripts/verify_frozen_gen3.py
```

No learning or production deployment is performed by the benchmark scripts.

`python scripts/run_gpu_comparison.py` runs the four measurements serially and
samples GPU utilization/physical memory once per second (initialization included).
It refuses to replace existing timing JSON. `profile_gpu_generation.py` is a
separate, instrumented CPU profile; never run it during a throughput comparison.
`plot_gpu_generation.py` renders throughput, move-latency CDF and actual batch-size
CDF. GPU event intervals include dispatch gaps and copies queued on that stream;
they are not a hardware-kernel-only occupancy measurement.
