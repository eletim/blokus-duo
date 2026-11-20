# Data scaling A/B/C — fixed before generation and strength results

This extends the PR #1 small pilot; its artifacts remain immutable. Production main/Frozen gen3 remain unchanged. Three architectures and frozen candidate policy exactly match the pilot.

## Data and learning

Generate 3,072 new training games and 192 independent development games using the three frozen gen3 policies plus H (width4, maximum6,500ms), both seats, random8ply legal starts. Eight CPU workers on distinct physical cores generate data; timed strength evaluation is sequential with no concurrent training/audit. Per-game files are atomic and resumable; wall-time-limited teachers can vary with hardware load. This is fixed-policy terminal-WDL supervised training, not iterated policy improvement.

Nested training sets: 384 / 1,536 / 3,072 games (4x /16x /32x prior96). Each contains equal numbers from all three seed teachers. All9 models per size use the same games, labels, target, order seed, optimizer and updates. Initialize every size from original frozen weights, not a smaller trained checkpoint. Train value-side conv/compression/value and A/B additions; policy is separate and immutable. Adam1e-4, batch64,16epochs, game-uniform MSE +0.1 original-value regularization, no dev-based checkpoint selection. Record train/dev errors each epoch. Thus larger sets entail both more distinct games and more updates, not an equal-update causal isolation. Dev games never become train data.

Exclude known historical evaluation states, previous pilot training/dev/evaluation visited states, original gen3 training index from adopted labels. No human games. Report exclusions and independent game counts. Do not count the27 models as27 independent datasets. Keep source/model/data SHA.

## Evaluation and decision

Before any strength results, freeze all models and three disjoint new banks; starts separated from all training/development and historical exclusions. Stage384 and1536: shared6-prefix curve bank, A/F,B/F,C/F; each36games (both seats,3seed). Descriptive only, never choose a checkpoint or size from these results.

Final3072:16-prefix validation bank, A/F,B/F,C/F,A/C,B/C;96games/pair,480total. F is original gen3+H. Same width4,max6,500ms/nativeCPU/torch1thread. Record actual latency,completeddepth,fallback. Only final3072 A or B eligible for independent confirmation when F point>50%, positive score diff, C point>=50%, >550ms nonpass fraction<=1%. Choose higher F point, tieA. This screening is intentionally permissive; no strength claim from validation. Never promote C or smaller checkpoints post hoc.

If eligible, new32-prefix confirmation bank, selected/F and selected/C,192games each. Bootstrap by prefix (both seats and3seeds jointly),20kreplicates,97.5%two-sided intervals for two confirmatory point comparisons. Adoption recommendation requires lowerpointCI>50% and positive mean score diff in both comparisons, plus <=1%moves>550ms. Report scoreCI, seeds and depth. No rescue tuning or sample addition. If no eligible model, confirmation bank remains unused. Report cannot establish universal futility or human superiority.

All600 prior pilot games remain separate. New generated3,264 plus216 curve and480validation games =3,960 before optional384confirmation. Independently replay all new games, audit scores/pass/legal moves, split/hash separation, arithmetic, frozen weights, fresh-clone CPU tests and exports. Preserve failure/non-adoption results. Update existing draft PR; do not merge/deploy.

## Interpretation

Look for train-vs-dev error and paired same-data A/B-vs-C differences across sizes; development accuracy is not playing strength. Failure may reflect limited games, WDL continuation labels, fixed top4 candidates, small model/representation, regularization or coarse spatial bins. No claim of convergence solely from16epochs. Measure actual learning curves and state these limits.
