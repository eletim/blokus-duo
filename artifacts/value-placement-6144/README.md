# 6,144-game placement-aware value experiment

Read [PLAN.md](PLAN.md) before running. This continues the [3,072-game study](../value-placement-scale/REPORT.md) without changing its saved results or models. Artifacts belong to research PR #1; no main merge or deployment.

From repository root, use the existing project Python environment with torch, numpy and built native legal generator. CUDA is used for training; exports work on CPU. Set `PYTHONPATH=.`. The original training-state index is restored as described in the prior study, SHA256 `cdd1f0f1ac7a0e722a670fb541ce904187949c6982cd03fda1fa7acad30f7b4c` (NumPy array content).

Recorded outputs must not be overwritten. Reproduction should use an isolated checkout and output tree, retaining the committed original artifacts.

```
python scripts/placement_more.py prepare
python scripts/placement_more.py generate
python scripts/placement_more.py features
python scripts/placement_more.py train
python scripts/placement_more.py banks
# Freeze models/banks before timing; checks must not overlap timed games.
python scripts/audit_placement_more.py checks
taskset -c 0 python scripts/placement_more.py validation
# Only if selection.json.passed is true:
taskset -c 0 python scripts/placement_more.py confirmation
# After all timing processes stop:
python scripts/audit_placement_more.py
python scripts/report_placement_more.py
```

Generation reuses the original teacher, random seed scheme and eight pinned cores, with new game IDs1024–2047 per teacher. Old training data are rows0..84360 of the combined feature file; new rows follow. Development data remain in `artifacts/value-placement-scale/dev-features.npz`. All6models start from original gen3 and train16epochs; no match-selected stopping.

The frozen P/Q references are the previous n1536/n3072 A16 models. Aliases are resolved by `placement_more.agent`; checkpoint architecture is loaded explicitly, so P/Q are never treated as new model types.

Validation uses the same16prefixes for P/F,Q/F,A/F and direct A/Q,A/C,C/F. `paired-curve.json` summarizes prefix-matched differences with descriptive95%intervals. Confirmation, if selected, uses a different32prefix bank and3primary-comparison correction. Validation intervals are descriptive, not six-comparison confirmatory claims.

Every timed game uses500ms,max6ply,width4,CPU native,torch1thread. Actual timing depends on machine/load and is recorded. Timing-limited generation and search are not bit-reproducible across machines. The replay audit checks actual saved games, targets, split hashes, data metadata and fixed source/model manifest. Same-state hash separation does not exclude all symmetries or strategic similarity.
