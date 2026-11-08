# Frozen H terminal-outcome value pilot: no confirmed improvement

The new value head is **not adopted**. Public H, main and Frozen gen3 were not changed. This was supervised terminal-WDL fitting, not an on-policy PPO update. No human game was used or included in this branch's new artifacts.

## Fixed experiment

Original gen3 seeds 1751401/2/3, original width4/max6 H, 500ms. Each seed generated32 training and8 held-out complete games (120 total), after six random legal opening plies. All later nonterminal eligible states use their actual game's terminal win/draw/loss from the current side's perspective. Both seats are represented: 480/460,479/459,479/464 training states per seed. Games receive equal total weight, not equal weight per state. There are940/938/943 training labels and231/236/239 held-out labels.

Only the existing value MLP was trained, retaining all policy, convolutional, global, position and prior parameters. CPU Adam1e-4, batch64,8 epochs,120 updates per seed, game-weighted MSE plus0.1 squared drift from the original predictions. No endpoint/seed selection. PLAN committed before generation at985a02e; endpoints and evaluation banks frozen atf0786ed.

Held-out game-weighted MSE decreased for all seeds:

|seed|original|trained|
|---|---:|---:|
|1751401|0.48746|0.46353|
|1751402|0.53611|0.53132|
|1751403|0.66209|0.61474|

These estimates use only8 held-out games per seed and are not evidence of improved playing strength.

## Validation

New12 random legal opening prefixes, paired seats and all3 fixed seeds:72 games. Candidate+H versus original gen3+H, same nominal500ms. Point=(wins+0.5draws)/games. Bootstrap groups are prefixes, keeping both seats and all seeds together. One primary comparison;95%CI below (the inherited aggregation also stores a conservative two-comparison interval, which is not required here).

- 30 wins,6 draws,36 losses: **45.83% point**,95%CI39.58–52.08%.
- Mean score difference **+0.278**,95%CI−1.542–+2.208.
- Per-seed points39.58%,54.17%,43.75%.
- Predeclared gate (>=55% point,positive mean score,at least2 seeds>=50%,p99<=750ms) failed. No confirmation games, rescue adjustments or public deployment.
- Candidate mean243.55ms vs original241.75ms; nonpass6ply completion82.43% vs82.46%. All nonpass searches finished at least4ply. Reported depth0/fallback counts are forced-pass calls, not lost nonpass searches.
- Candidate max525.73ms, original501.64ms. The500ms budget is soft, not a latency guarantee.

The small pilot did not establish improvement; it also does not prove that terminal-outcome learning in general is ineffective or that this model is inferior. The positive score difference and negative point estimate must both be retained.

## Verification and costs

- Independent coordinate-based legality/pass/final-score replay passed for all120 generation and72 validation games.
- Recorded selected labels were checked against actual turns, terminal outcomes and state hashes. Historical training-index checks passed for selected labels and evaluation continuation states; index file SHA recorded in audit.json. Prior evaluation starts were excluded. Train/dev/validation visited-state intersections were0 after the shared opening phase.
- All non-value tensors are unchanged. Nine sampled full-forward policy logit comparisons were bit identical on CPU.
- Full120 updates, final value parameters, Adam state and RNG per seed reproduced exactly; stored exports match. Independent-process10 updates equals4+resume6. CPU inference is exercised by all evaluation games.
-16 relevant value/search tests passed. Existing production files were not modified; this is not a fresh production deployment validation.
- Generation:332.20s elapsed with3 concurrent CPU processes. Training time was not separately instrumented. Strength evaluation:484.95s, one process, without training/audit overlap. Independent audits ran afterwards.

## Limits and next hypothesis

Only32 training games per seed, a frozen representation, a single small value head and the same policy-filtered width4 search were tested. Generation uses a time-limited H, so machine load can affect completed depth and generated games; saved trajectories, rather than regenerated games, define the training dataset. Initial opening random seeds are shared across seeds. Actual terminal outcomes describe these frozen H continuations, not minimax truth.

Policy imitation and this value-only pilot have both failed to improve public H. A future joint policy/value self-play experiment is a separate hypothesis, not an achieved result. Search-selected actions must not be passed to an on-policy PPO loss as if sampled from that policy. Human playing strength remains unestablished. No additional learning or confirmation is running.

## Reproduction

Use this branch's existing Python dependencies (CPU torch is sufficient). Preserve source models in artifacts/ppo-gen3. Saved games, labels, final inference models, optimizer checkpoints, fixed banks, hashes, timing records and summaries are in this directory.

```
python scripts/check_h_terminal_resume.py endpoints
python -m pytest -q tests/test_value_leaf.py tests/test_ppo_deeper.py tests/test_deep_search.py
python scripts/play_h_distill.py --checkpoint artifacts/h-terminal-value/student-1751401.pt --model-seed 1751401 --search --human-seat 0
```

The generic research play script also loads these value-only exports despite its legacy distillation filename. `endpoints` reconstructs updates from saved selected labels without requiring the external historical training index. Full exclusion/audit regeneration requires the archived historical index at the location and SHA recorded in audit.json; that index is not included in this branch. Do not overwrite the completed artifacts to rerun an experiment. A timing rerun requires a separate output directory and can vary with runtime load.
