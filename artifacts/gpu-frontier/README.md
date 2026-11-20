# Reproduce the fixed-teacher ready-frontier comparison
Use the repository environment with native observations built (`python -m
blokus.ppo_native build`), CUDA torch and the unchanged gen3 exports.

```bash
export PYTHONPATH=.:scripts OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python scripts/check_frontier.py
python -m pytest tests/test_suspended_frontier.py
# Existing bank is fixed. Do not run prepare over it.
python scripts/run_frontier_comparison.py
python scripts/audit_frontier.py
python scripts/check_frontier_resume.py
```

Completed timings are immutable: use a clean artifact output directory for a new
measurement, retaining the original bank/model hashes. Source scripts use
`artifacts/gpu-frontier`; prior synchronous-RPC artifacts are never overwritten.
Generation requires the historical exclusion union reconstructed by
`python scripts/gpu_generation.py rebuild-exclusions` and the referenced original
training index; these are provenance/audit dependencies, not inference dependencies.
Game records are atomic gzip JSON. Complete records are skipped after validating
the task identity. Incomplete files never become terminal reward labels. A restart
replays the interrupted game from its saved initial Python RNG, not from a serialized
mid-node stack. CPU reference remains available through `frontier_generation.py cpu`.

The standalone `FrontierSearch` accepts normal frozen CPU/native or Python policy
observations; `FrontierEvaluator` provides CPU or CUDA inference. `run_frontier`
schedules generators without any model training. The production search is unchanged.

To repeat without overwriting committed measurements, choose unused repeat IDs:
```bash
python scripts/frontier_generation.py cpu --repeat 4
python scripts/frontier_parallel.py --repeat 4
```
These process the same fixed192-game bank with CPU8 and GPU64-frontier cohorts.
Do not run them concurrently. `audit_frontier.py` deliberately audits the frozen
formal repeat1/2; adapt a copy for separately named future repetitions.
Initialization measurements cover process/model setup and model warmup after
Python module import. The sampled outer-process telemetry also includes imports,
but its one-second polling is not an exact process startup measurement. Neither
scope should be presented as a pure GPU-kernel benchmark.

Validation note: the first new fallback test chose a forced-pass fixture and tried
to read nonexistent root scores. That test log is retained. The fixture was changed
to a nonpass middle position; this did not alter search code or measured games.
A post-measurement constructor guard explicitly rejects custom value evaluators,
which this frozen-teacher scheduler does not support; the measured default path
is unchanged. Mid-search stack serialization and arbitrary learned value heads
are not supported by this experimental implementation.
