# Value placement A/B experiment — pre-evaluation plan

Base: main d8f50664a8fb6afaa3b6f8775e7f212e66fc14d7. Frozen gen3 policy and model files are unchanged. No human archive data. No production deployment or main merge.

## Hypothesis and controls
A adds 252 exact-mobility features: for both players and each of 21 pieces, legal availability, log(1+count)/log(4097), and four centroid-quadrant fractions. The new linear projection enters the value hidden layer; zero initialization preserves original outputs. Counts enumerate actual legal placements, not frontier points. This describes current opportunity, not guaranteed future placement.
B applies remaining-piece-conditioned FiLM to both 8-channel convolutions, before ReLU and before the 32-dimensional compression. Zero modulation preserves original outputs. This does not guarantee exact legality understanding.
C copies and retrains the original value architecture, including convolution and board compression. F is untouched gen3 + H.
Policy is a separate unchanged gen3 instance for all conditions, including opponent nodes. Only value is replaced. Parameter-count and compute differences are limitations; this is not a parameter-matched architecture attribution experiment.

## Finite generation and learning
Cost-only pilot: random legal game seed 58600001, positions 8/16/24, complete 4ply. No playing-strength selection. Native legal scan for all methods; CPU inference, one torch thread.
Generate 32 train and 8 development games with each of 3 frozen gen3 seeds (120 total), unique seed-specific random 8ply starts, then H/500ms self-play to actual termination. Include both players' states after prefix, not just even plies. Shared pool for all 9 learners. No teacher-search value labels: use actual terminal win/draw/loss from the state player's perspective. This remains an imperfect continuation policy, not minimax ground truth.
Exclude original gen3's 27,533,566 visited states and historical evaluation starts from learning. Check train/dev state separation. Keep exclusion metadata and full games. Initial-prefix states are inherently shared near the empty board and are not learning/evaluation samples.
Train A/B/C from each seed with Adam lr 1e-4, 16 epochs, batch64, game-uniform loss weights, WDL MSE + 0.1 MSE to the original value. Same example order and update count per seed. Fixed final epoch, no dev checkpoint selection. Train copied value trunks; source policy is immutable. Save inference exports and optimizer/RNG state. CPU/CUDA initialization, source isolation, legal features, exact-tree search, and resume checks required.

## Playing-strength comparisons
Freeze all code/models/banks before timing; run sequentially without training/audits concurrently.
Validation: fresh 12 prefixes, lengths8/10/12, both seats, all3 paired seeds =72 games each for A/F, B/F, C/F, A/C, B/C (360 games). H width4,max6,500ms, native CPU, same environment. Record elapsed time, completed depth and fallback. Identical random starts across comparisons; aggregate by prefix, not independent seeds/games.
Fixed-tree diagnostic: separate 4 prefixes, both seats,3 seeds, same5 comparisons =120 games; width4,max4,budget60s. Every nonpass move must finish4. This controls tree depth, not equal computation. Never use this diagnostic to rescue a failed practical gate.
Gate, chosen before any results: A or B eligible only if versus F point>=55%, positive mean score difference, at least2/3 seeds point>=50%; versus C point>=50%; fewer than or equal1% of nonpass moves exceed550ms. Choose higher point vsF, tie A. No post-hoc promotion of C.
If eligible, independent confirmation uses new32 prefixes, both seats,3seeds=192 games each versus F and C (384 total). Validation is never pooled into confirmation. Prefix-cluster bootstrap20,000 replicates, fixed rng771, 97.5% intervals (Bonferroni for two comparisons). Superiority claim requires point interval lower>50% and score interval lower>0 versus F; representation-specific improvement requires the same against C. Otherwise no adoption. Validation intervals are descriptive, not proof.
If gate fails, stop without extra labels, epochs, depths, sample-size changes, or confirmation. Fixed-tree diagnostic still completes as planned.

## Audits and deliverable
Independent coordinate legality/pass/score replay for generated and evaluated games, state separation and original index checks, source/model SHA, deterministic save/resume, CPU loading and fixed-policy tests. End with branch/PR and report including failures and limitations. Frozen tag/weights and production main remain unchanged.
