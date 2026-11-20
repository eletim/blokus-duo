# A value learning: 3,072 to 6,144 games

Plan fixed before generating new data or inspecting new strength outcomes.

Question: does more distinct selfplay data improve legal-placement-aware value beyond the confirmed 3,072-game A? Prior 384/1,536 curve matches had only six prefixes; they do not establish a win-rate plateau. More epochs alone showed development overfit. This trial increases unique games, not epochs.

- Preserve main, Frozen gen3, previous artifacts and Web. Existing research PR only.
- Teacher unchanged: three frozen gen3 models with H, width4, max6ply,500ms. New 3,072 games (1,024 per teacher), random8ply then teacher to terminal. Combine the prior3,072 training games; use prior192 development games unchanged. No human games.
- Exclude original training index and historical/current evaluation states and development states from new generation/labels. Keep old train intact; test splits must have no overlap with combined train/dev.
- A and C only, all3seeds, start original gen3 anew. Same previous settings:16epochs,Adam1e-4,batch64,terminal WDL game-weighted MSE +.1 original-value penalty. Frozen separate policy. No checkpoint selection based on new tests. Record every epoch; no automatic extra epochs.
- Reference P=previous1,536-game A16; Q=previous3,072-game A16; A=new6,144-game A16; C=new6,144 original-value structure; F=unchanged gen3+H. All models fixed before validation.
- Validation common new16prefixes,8/10/12 random plies,both seats,3seeds: PF,QF,AF,AQ,AC,CF =576games. Shared prefix/seat/seed for P/Q/A versusF, including paired prefix bootstrap differences (descriptive95%CI). Main progression screen: A/F and A/Q point>50% and score difference>0; A/C point>=50%; all three A nonpass>550ms rates<=1%. Decisions from integer win/draw counts. No choosing another variant/epoch on failure.
- If screen passes, independent new32prefixes,AF,AQ,AC =576games. At each pair192games;3primary comparisons, prefix-cluster bootstrap20,000 draws,98.3333%two-sided intervals (Bonferroni3). Require all lower point bounds>50%,positive mean score and>550ms rate<=1%. Do not pool validation and confirmation. Confirmation bank fixed before strength outcomes.
- Same500ms ceiling, width4/max6, CPU native, torch1thread, core0, games sequential with no heavy training/audit. Record actual time/depth/fallback. Same real compute time is not promised.
- Save exports,resume state,data hashes,per-epoch train/dev errors,all games. CPU load/resume and independent coordinate replay audit. Recompute summaries and split checks. No extra samples or rescue tuning if result fails.
- Distinct data also increases gradient updates; no isolated data-vs-update causal claim. Model capacity differs A/C. Frozen-teacher data and random starts do not prove human/empty-board superiority. A flat/noisy finite curve does not prove a universal strength ceiling.
