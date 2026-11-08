# Reproduction

Use this research branch and its PLAN.md/fixed-shas.json. Runtime:Python3.12,Torch2.7.1+cu128,RTX5090 for training;CPU/native for evaluation. Inference exports and evaluation JSONs are self-contained in artifacts/joint-selfplay. Human logs are neither required nor included.

Training script requires the original gen3 B iteration5120 optimizer checkpoints and historical training-state index at its documented paths. It refuses to overwrite history segments. A complete rerun must use an isolated checkout/output directory; do not silently reuse cached evaluation files for a new model.

```bash
python scripts/joint_selfplay.py audit_training
python scripts/joint_selfplay.py audit_evaluation
python -m pytest -q
```

The training and bank preparation commands are run_training and prepare, respectively; evaluate reuses a saved result if that seed/phase file already exists. Saved PLAN/training-size/models/fixed-shas and all histories/checkpoints/rollout RNG state preserve the exact run. To inspect a saved candidate without changing defaults:

```bash
python -m blokus.ppo_play --device cpu --backend native --checkpoint artifacts/joint-selfplay/student-1751401.pt --model-seed 1751401
```

Candidate weights are research-only: independent confirmation failed, and they are not a public gen4 release. Main Web continues to use original gen3 with existing deep search selected initially.
