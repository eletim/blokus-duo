# Joint PPO continuation: independent improvement not established

## Result and disposition

The finite joint policy/value PPO continuation failed the predeclared independent confirmation gate. Do not publish these weights as gen4, select seed1751403 post hoc, extend this confirmation, or describe validation as confirmation. Original public/Frozen gen3 weights remain unchanged.

| phase | games | W/D/L | point (win1,draw0.5) | prefix-cluster95% CI | mean score difference | score95% CI |
|---|---:|---|---:|---|---:|---|
| validation |72|40/4/28|58.33%|50.00–65.97%|+1.68|−0.85–+3.69|
| independent confirmation |192|98/10/84|53.65%|47.14–60.16%|+0.82|−0.20–+1.82|

Validation passed the selection rule. Independent confirmation did not satisfy either pooled lower bound. The preselected public seed1751401 also failed its point/score condition (29/4/31,48.44%,−0.58). Other seeds:1751402 33/3/28,53.91%,+0.77;1751403 36/3/25,58.59%,+2.27. Seed variation is reported without selecting the best model. These results do not prove equivalence or no possible benefit.

## Training and scope

All three matching original gen3 iteration5120 checkpoints were restored with their optimizer. Trainable policy, value and shared NN parameters were jointly updated; the heuristic prior buffer was unchanged. Fresh sampled categorical on-policy rollouts used terminal WDL, never search-overridden actions. The opponent mix was25%current selfplay,25%past snapshot,25%frozen original gen3,25%heuristic, with balanced learner seats. Fixed LR1e-4, minibatch64, existing2 PPO epochs/KL stopping/clipping/entropy/gradient limit.2048 iterations×16 games=32768 games per seed,98304 formal games total. Cost pilot128 games was separate. The size was chosen from cost before strength evaluation; no checkpoint/seed selection by performance. Formal training wall1006.93s (16.8min). No human game logs were used.

This finite configuration does not isolate the effects of LR, opponent mix or training length. Lack of improvement does not identify one cause. Small prior policy-only and value-only failures remain preserved. Further tuning was not conducted after this confirmation result.

## Matched evaluation

Both models use the existing H width4/max6 completed-tree search,500ms cap,CPU/native observation backend as available in the deployed Web. Native source and models/banks were hashed before evaluation (fixed-shas.json). Earlier Python-backend experiments are not a matched timing baseline. New6–10-ply legal-random prefixes:12 validation,32 confirmation, each×both seats×3 fixed seeds. Bootstrap units are prefixes (six correlated games per prefix),20000 bootstrap resamples. There is one primary candidate comparison; the extra Bonferroni interval emitted by shared tooling is supplementary. Do not combine validation and confirmation. The claim is limited to these fixed seeds and prefix distribution, not human superiority or arbitrary openings.

Confirmation mean elapsed time: candidate114.81ms,original117.62ms per turn. p99:310.71/321.01ms; maxima392.73/397.82ms. Both completed6ply on all non-pass moves; no500ms overruns observed. Forced passes appear as depth0/fallback in shared aggregate output and are not timeout failures. Equal cap does not guarantee equal wall consumption or future strict latency. Timing wall: validation250.00s,confirmation623.39s (total14.56min).

## Verification

Exact separate-process8 versus4+resume4 CUDA continuation matched model,past model,Adam and all RNG states; generated game hashes matched. All98432 formal+pilot games were checked for legality, scores, terminal label sign and learner seat. Independent coordinate replay sampled385 games (not all training games); action index checks sampled every256 learner targets. All1953455 targets and3393633 unique visited states are recorded in training-audit.json. New evaluation starts and visits are checked against original and new training indices; validation/confirmation visited-state separation is also checked. All264 evaluation games passed independent coordinate legality/score replay and recomputation of saved aggregates/CIs. No evaluation visits overlapped either training index, and validation/confirmation visited states were disjoint. Full research suite:116 tests passed in23.34s. See evaluation-audit.json and tests.log.

## Public change, separate from this failed candidate

Previously the browser initially selected policy-only gen3 despite the already validated deep-search option. Main commit d8f5066 changes the initial choice to existing deep H at500ms. Existing model,search,mode choices and archive behavior are preserved. Isolated browser regression covered both seats,rotation/flip,legal placement,AI response,terminal game,download,mode switching,stale response rejection,session isolation,mobile width and no JS errors. Production smoke used test-kind archive records. HTML is read for each request,so no service restart was needed. See main artifacts/deep-default/CHECK.md. This is an exposure/default fix, not a new learned-strength result.

An untested gen4 Web integration draft was prepared while confirmation ran. It is ineligible for publication because the gate failed. It is preserved locally as research/joint-web-draft-not-adopted@9702004 (untested) and is not merged or deployed. Frozen tag gen3-frozen-2026-11-06 and existing model SHA256s are unchanged.
