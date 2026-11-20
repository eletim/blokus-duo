# Fixed 24,576-game research run

This directory is a new run, not a replacement for `value-placement-scale`,
`value-placement-6144`, or `gpu-async512`. Main, Frozen gen3, and the Web are
unchanged. Only the newly generated games are archived here. Old 6,144 games
and original PPO state index are referenced by SHA in the data/bank manifests.

## Dependencies and fixed inputs

Run at repository root, Python3.12+, NumPy2.5.4, PyTorch2.7.1+cu128 on RTX5090.
Use the repository environment and native extension, not a new teacher:

```bash
export PYTHONPATH=.:scripts OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
PY=/home/eletim/blokus-duo-ai/.venv/bin/python
$PY -m blokus.ppo_native build
$PY scripts/verify_frozen_gen3.py
```

The public gen3 inference weights remain unchanged. They are **not** the new
value checkpoints. New A/C value copies are trained on real terminal win/draw/loss;
their policy is the separate original gen3. A adds mobility features and C does
not. The original-value penalty is 0.1, as in the preceding experiment.

Restore the fixed new bank with `gzip -dk artifacts/value-placement-24576/bank.json.gz`.
`bank-manifest.json` records its uncompressed SHA. The original training index
required by the exclusion audits is `placement_experiment.INDEX`, array SHA
`cdd1f0f1ac7a0e722a670fb541ce904187949c6982cd03fda1fa7acad30f7b4c`.
The historical source paths/hashes are recorded in `bank-manifest.json`;
`data-manifest.json` references the old feature file without duplicating it.
The new feature files are the exact arrays used for both A/C training.

## Generation, audit, and real training

The completed run must not be overwritten. In a fresh checkout, after restoring
the bank and historical audit inputs, these are the actual stage commands:

```bash
$PY scripts/async_generation.py --bank artifacts/value-placement-24576/bank.json \
  --out artifacts/value-placement-24576/games \
  --concurrency 512 --batch 128 --fast-encoding --graphs
$PY scripts/placement_large_features.py
$PY scripts/check_placement_large.py
$PY scripts/placement_large_train.py
$PY scripts/check_placement_large.py
$PY scripts/check_large_exports.py
$PY scripts/placement_large_eval.py banks
$PY scripts/placement_large_eval.py validation
# Only when selection.json passed is true:
$PY scripts/placement_large_eval.py confirmation
$PY scripts/audit_placement_large.py
```

Generation retains completed per-game records and regenerates only unfinished
games. To inspect rather than regenerate, extract the flat game files from each
`game-archives/*.tar.gz` into `games/`; the archive manifest verifies every byte.
Training restores each latest Adam/RNG checkpoint and resumes at the next epoch.
Do not discard `resume-*.pt` when resuming. Running training against this completed
directory does not perform new updates. To reproduce training from scratch without
overwriting evidence, create a separate output directory and copy only PLAN.md and
data-manifest.json there (the manifest retains the original, hashed feature paths):

```bash
mkdir -p artifacts/value-placement-24576-reproduction
cp artifacts/value-placement-24576/{PLAN.md,data-manifest.json} artifacts/value-placement-24576-reproduction/
$PY -c "from pathlib import Path; import placement_large_train as t; t.R=Path('artifacts/value-placement-24576-reproduction'); t.train()"
```

This optional command runs the full finite six-model training; it is not needed
for CPU inference or for verifying the saved exports. A/C share one epoch selected only by
mean new-dev MSE across all six models; selection and evaluation bank hashes are
fixed before strength games. No model/seed selection uses match outcomes.

The old 6,144 CPU-generated games used a500ms ceiling; new games use completed
6ply. Known GPU near-tie rounding differences remain. This is therefore an
expansion experiment with stated generation differences, not bit-identical data
or an isolated proof that quantity alone causes any gain.

## History-free CPU inference

Only code, original gen3, and the chosen small A/C export are needed:

```bash
$PY scripts/placement_value_play.py \
  --checkpoint artifacts/value-placement-24576/A-1751401.pt --backend python
$PY scripts/placement_value_play.py \
  --checkpoint artifacts/value-placement-24576/A-1751401.pt --backend python --human-seat 0
# --human-seat 1 for second player; --backend native after building the extension.
```

The CLI verifies the original-policy SHA. It defaults to width4/max6/500ms CPU
search, retaining only completed search stages. This is a research candidate,
not a new public gen3 generation or a claim of strength against humans. Inference
exports omit Adam/RNG; `resume-*.pt` are distinct training continuation files.

## Regeneration caveats

Do not rerun timed matches to select a better outcome. CPU deadline completion
can vary with load; fixed code/model/bank hashes and recorded real timings are the
reference. `PLAN.md` is the preregistered decision rule. Loss curves are not proof
of playing-strength gain. All substantive results and limitations are in REPORT.
