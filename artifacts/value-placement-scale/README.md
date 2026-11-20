# Placement-value data scaling

See [PLAN.md](PLAN.md) for the plan fixed before generation. This is an extension of [the original small pilot](../value-placement-ab/REPORT.md); those original artifacts remain unchanged. Results and limitations will be recorded in REPORT.md after evaluation, not inferred from training loss.

Run from the repository root with Python environment from pyproject/requirements, torch (CUDA for training), numpy, and native legal generator built. Original frozen models are in artifacts/ppo-gen3. Use `PYTHONPATH=.`. No production endpoints or human game archives are accessed.

The original training-state index is required at `/tmp/ppo-lr64-training.npy`. If absent, restore with `scripts/restore_placement_index.py` per the original pilot README; the index is reconstructed from committed historical state hashes, not downloaded from an unstated source.

Commands (outputs are committed; do not overwrite a completed run):

```
python scripts/placement_scale.py prepare
python scripts/placement_scale.py generate
python scripts/placement_scale.py features
python scripts/placement_scale.py train
python scripts/placement_scale.py banks
# Commit fixed models/banks before any strength comparison.
python scripts/placement_scale.py curve --size 384
python scripts/placement_scale.py curve --size 1536
python scripts/placement_scale.py validation --size 3072
python scripts/placement_scale.py select
# Only if selection.json.selected is A or B:
python scripts/placement_scale.py confirmation --size 3072
# Stop all timing runs before tests/audit:
python scripts/audit_placement_scale.py
python scripts/check_placement_scale.py
```

Generation uses eight pinned physical cores, per-game atomic gzip checkpoints. It skips only already completed files, and the final audit replays all files. Wall-clock-limited generation/search are not bit-reproducible across machines/load. Random seeds, full moves, outcomes and model/source hashes are saved.

`train-features.npz`/`dev-features.npz` preserve actual learning inputs. Array x0=board4x14x14,x1=state49,x2=mobility252,x3=WDL,x4=game weighting1/labels-in-game,x5=game id within source seed. Labels JSON maps each row to source seed/game/ply/hash. Stage N includes ids<N/3 from each of three teachers. All models at a stage use identical rows. x4 is normalized to mean1 for optimization; error reports use game-weighted means.

Each model starts anew from its corresponding frozen gen3. A adds mobility features; B applies inventory FiLM; C retains original architecture. Conv/compression/value-side parameters are learned, candidate policy is separately frozen. `resume-*.pt` includes final Adam and NumPy permutation RNG. No dropout/random augmentation is used. Frozen policy weights, source data SHA, updates, and per-epoch train/dev errors are recorded. Saved exports are CPU-loadable.

Lower-stage curve games are descriptive. Final validation selects at most one of final-stage A/B; only its separate confirmation can support a superiority recommendation. No post-hoc checkpoint, seed, size, hyperparameter choice. See PLAN for exact gates and confidence intervals. A/B/C parameter counts differ, and larger datasets also produce more updates: interpret the controls accordingly.

## Completed larger study

See [REPORT.md](REPORT.md) for the scaling results and the separate development-selected stopping experiment. Fixed-16-epoch A passed independent confirmation against both frozen H and the same-data original-architecture control. The later stopping experiment uses a different, predeclared bank and is reported separately; it is not a direct A6-versus-A16 comparison.

The separate stopping study follows [EARLYSTOP_PLAN.md](EARLYSTOP_PLAN.md):

```
python scripts/placement_earlystop.py train
python scripts/placement_earlystop.py banks
# Commit the models and bank before playing:
python scripts/placement_earlystop.py check
python scripts/placement_earlystop.py evaluate
# After all timing runs stop:
python scripts/placement_earlystop.py audit
python scripts/report_placement_scale.py
```

Epoch choices are derived from the recorded mean development MSE across all three seeds. A/B/C stop at epochs 6/5/7. Do not use match results to select epochs or seeds. Saved results include both successful and unsuccessful conditions. No main merge or Web deployment is part of this study.
