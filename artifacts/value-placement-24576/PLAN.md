# GPU-generated data expansion to 24,576 training games

Fixed before generation, training, or new strength outcomes (2026-11-20).

## Scope and data
- Keep main, Frozen gen3, Web and all prior results unchanged; same research branch/PR only. No new speed experiment or optimization sweep.
- Use the verified async512 implementation: 512 slots, 8 CPU workers plus coordinator, batch cap128, native fast encoding, float32 CUDA graphs. Frozen gen3 seeds1751401/2/3, policy width4 and completed2/4/6ply. No learned-A teacher.
- Add18,432 training games (6,144 per teacher) to the existing6,144 games. Generate a separate new192-game development bank (64 per teacher). Random8ply starts; both turns and all subsequent stages provide actual terminal-WDL targets. Prefix states are not labels. No human logs.
- Fixed independent per-game RNG series 885200000000 + seed_index*100000 + game_id; dev uses885300000000. Do not reuse speed/throughput banks, old evaluation, old development, or original PPO training states. New bank and model/source SHAs fixed before generation.
- Existing6,144 CPU games used500ms H and may stop below6ply. New games require completed6ply and GPU rounding may alter near ties. We preserve this provenance; the comparison is data expansion with this known generation difference, not a perfectly isolated data-count causal claim.
- All new visited states are checked against historical/old training/dev/evaluation and original training index. New train/dev starts and visited states must be disjoint. If a collision or malformed game is detected, stop and document it; do not silently drop labels or change the sample after outcomes.

## Learning (real CUDA updates)
- A and C, all3 seeds, each initialized afresh from its corresponding Frozen gen3. Policy remains a separate frozen gen3; value-model copies are trainable. A adds legal-placement mobility features, C retains original value structure. A's residual starts at zero, so initial predictions agree.
- Same combined24,576 games and identical examples/order/update budget for A/C; game-normalized terminal-WDL MSE plus0.1 squared deviation from original gen3 value. Adam1e-4, batch64, float32,16 epochs maximum, no LR/architecture rescue or added epochs. Record actual examples, updates, weights, train/dev curves and wall.
- Save every epoch's inference weights and last optimizer, NumPy/Torch/CUDA RNG, configuration/data hash. Epoch-boundary restart must reproduce uninterrupted updates; completed epoch outputs must not be overwritten inconsistently.
- Select ONE COMMON epoch for all six models by minimum mean new-dev MSE across A/C and3seeds among epochs1..16; exact ties choose earliest. Thus A/C selected update counts remain equal. No evaluation-result-based checkpoint/seed selection. Epoch0 is a recorded baseline only.
- Old3072A/6144A dev curves may be computed on this new dev bank descriptively; no new intermediate-size training arms. New dev selection is not independent strength evidence.
- Non-finite loss/gradient/weights stops the affected run and prevents promotion; diagnose rather than add conditions.

## Evaluation (after generation/training and heavy audits)
- Fixed newly separated validation16 prefixes and independent confirmation32 prefixes,8/10/12 random opening plies. Both seats, all3 corresponding seeds. A=new selected A, C=same-data selected C, Q=prior successful3072A16, F=Frozen gen3+H. Three pairs AQ/AF/AC:96 games each validation,192 each confirmation.
- CPU native, torch1thread, core0; all timed games sequential without generation/training/heavy audit. Same width4/max6/500ms ceiling. Actual mean/max/p95/p99 time, depth completion and fallback retained; not equal actual wall and not a human-strength claim.
- Validation screen: A/Q and A/F strictly above50% point AND positive mean game-score difference; A/C point>=50%; each comparison A's nonpass>550ms rate<=1%. Integer win/draw counts define boundaries. No rescue if failed; validation still completes all3 pairs.
- If screen passes, complete all three confirmation pairs on the fixed32prefix bank; no sample extension. Prefix-cluster bootstrap20,000 draws combining both seats/all3seeds, ordinary95% CI and Bonferroni3 family98.3333% CIs for point/score. Success requires each adjusted point lower bound>50%, positive mean score, and A>550ms rate<=1%. Do not pool old/validation/confirmation games or declare equivalence from nonsignificance.
- Bank/exports fixed before playing-strength outcomes; report seed/seat results and limitations even on failure. No main adoption in this task.

## Audit, bounds and reporting
- Independently replay every new generated/evaluation game (coordinate legality, forced pass, terminal bonuses/scores); check labels' actual-turn WDL, completed depth, source/model SHA, independent RNG, split disjointness, frozen policy, CPU export, real optimizer/RNG restart, aggregation arithmetic and needed regressions.
- Heavy generation, feature/audit processing, training and timed evaluation run in separate phases. No other service/run is stopped. Estimated new generation~60min from prior throughput; feature/audit~20–60min; six-model16epoch training~20–35min; validation~20min and conditional confirmation~40min. These are estimates, not guarantees. Finite18,624 new games/16epochs/16+32prefixes, no unlimited extension. Report delays rather than shrink the sample.
- Save new records only; reference old datasets/index by SHA, no duplicate old histories. Archive new trajectories in shards below Git's file-size limit. Save plots, exports, reproducible CLI, update PR and return measured strength as well as generation/training execution facts.
