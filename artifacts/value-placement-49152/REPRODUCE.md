# Finite data-scale extension: reproduction

Run from repository root. This is research only; no main/Web/Frozen writes. See PLAN.md for fixed sample/selection rules. Python3.12, NumPy and torch2.7.1+cu128 environment from the preceding experiment; RTX5090. CPU inference also works with ordinary CPU PyTorch. Native C compiler required for the measured native path; `--backend python` remains available for inference.

```bash
export PYTHONPATH=.:scripts OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
PY=/home/eletim/blokus-duo-ai/.venv/bin/python
$PY -m blokus.ppo_native build
# Existing bank is fixed; do not prepare or regenerate a completed bank.
# Restore the exact pretty-printed bank representation if needed (gzip uses compact JSON).
$PY - <<'RESTORE'
import gzip,json,hashlib
from pathlib import Path
r=Path('artifacts/value-placement-49152')
d=json.load(gzip.open(r/'bank.json.gz','rt'))
b=(json.dumps(d,indent=2)+'\n').encode()
assert hashlib.sha256(b).hexdigest()==json.loads((r/'bank-manifest.json').read_text())['bank_sha']
(r/'bank.json').write_bytes(b)
RESTORE
$PY scripts/monitor_generation.py --telemetry artifacts/value-placement-49152/generation-telemetry.json \
 "$PY" scripts/async_generation.py --bank artifacts/value-placement-49152/bank.json \
 --out artifacts/value-placement-49152/games --concurrency 512 --batch 128 --fast-encoding --graphs
$PY scripts/placement_extension_features.py
$PY scripts/placement_extension_eval.py banks
PLACEMENT_STAGE=artifacts/value-placement-49152/n36864 $PY scripts/placement_extension_train.py
$PY scripts/placement_extension_eval.py validation --stage 36864
PLACEMENT_STAGE=artifacts/value-placement-49152/n49152 $PY scripts/placement_extension_train.py
$PY scripts/placement_extension_eval.py validation --stage 49152
$PY scripts/placement_extension_eval.py confirmation
$PY scripts/audit_placement_extension.py
PLACEMENT_STAGE=artifacts/value-placement-49152/n49152 $PY scripts/check_extension_resume.py
```

Commands are stage operations, not a request to rerun completed work. The supervised `scripts/placement_extension_pipeline.py` starts only absent stages and preserves completed exports/match shards; its Village report IDs refer to this particular request. To reproduce elsewhere use individual commands and a separate output tree. The trainer resumes only an exact configuration/data/code hash match, from its saved epoch-boundary Adam plus all RNG states. It regenerates no completed models. Generation restores independently seeded unfinished games and preserves complete game files; the original configuration must match. Archive restore instructions and final source hashes are included with the completed archive manifest.

Old24,576 training files are SHA-referenced in stage data manifests, not copied. Prior dev and strength/throughput games are excluded from new training. Rebuilding the separation audit requires the historical index referenced in bank-manifest.json; inference needs none of it.

## History-free CPU inference

```bash
$PY scripts/placement_value_play.py \
 --checkpoint artifacts/value-placement-49152/n49152/A-1751401.pt \
 --backend python --human-seat 0
# Use --human-seat 1 for the other seat; omit for AI/AI smoke.
```

The CLI loads the unchanged separate gen3 policy and the research value export; it checks the policy source SHA. It does not need training games, development data, optimizer checkpoints or research indexes. Default search remains width4/max6/500ms, with only completed depth stages adopted. This is not a new public generation or a claim of strength against humans.

## Validation

```bash
$PY -m pytest -q tests/test_placement_extension.py
PLACEMENT_STAGE=artifacts/value-placement-49152/n49152 $PY scripts/check_extension_exports.py
$PY scripts/verify_frozen_gen3.py
```

Exports (`A-*.pt`/`C-*.pt`) are separate from restart checkpoints (`resume-*.pt`). All3seeds remain. Final confirmation never selects an epoch, seed or data stage. Timing of fixed500ms searches is sensitive to host load; no claim of bit-identical time-limited games between machines. GPUgeneration may differ at near ties fromCPU even with fixed completed trees, as documented by the preceding async512 validation.
