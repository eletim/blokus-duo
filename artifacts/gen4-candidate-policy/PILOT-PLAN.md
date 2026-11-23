# Fixed gen4 candidate-policy pilot (2026-11-23)

Base main2529f50 / Frozen gen4 ccc6d73. No public changes or value training.
Hypothesis: fixed gen3 top4 can omit useful roots under gen4 value. Search estimates are not minimax truth or actual WDL labels.

Audit: gen4 value is a separate network. TerminalNet policy head can update with conv/global_net/position/value/prior frozen. All gen4 value bytes remain fixed. Web remains on pinned checkout. GPU currently idle; CPU service active. Use four CPU workers at most for offline teachers, no timed evaluation overlap.

Pilot: new 2 games per seed (1751401/2/3), random8ply start then fixed gen3 greedy policy continuation, fully legal terminal histories. Extract paired ply12/13,20/21,28/29, classified early/middle/late and actual state.turn. If either pair member terminal or <=1 legal candidate, omit both; report counts. Independent game RNG, no shared arrival RNG.
Compare fixed root4/inner4/6ply, root12/inner4/6ply and root12/inner4/4ply using fixed gen4 value, no time cutoffs accepted (60s safety cap). Save roots, scores, moves, depth, elapsed, nodes, teacher agreement and omitted-top4 outcomes. This isolates root width at fixed depth; the 4ply control measures stability. CPU model/value and terminal/pass conventions identical to original search. Singleton root4 oracle equality must pass before pilot. Value tolerance 1e-7 (same CPU implementation expected exact).

This small bank is development only, never training or independent evaluation. After pilot cost, commit full PLAN before substantive diagnosis/training/evaluation. Target finite 2–4h cycle: diagnostic12 games/seed, training128 games/seed, dev16 games/seed (paired six positions), eight CUDA epochs, policy-head-only soft teacher distillation plus fixed prior-policy KL. Confirm detailed quantities and gate before full data. Final CPU500ms gen4 comparison all3seeds, swapped seats, fresh prefixes. If diagnostic disproves candidate-omission signal, report evidence and seek direction rather than manufacture a training success.

Prior evaluation/train exclusion index and new phase/game separation must be applied before full experiment; common initial empty boards are not treated as heldout roots. Existing async512 does not directly support root12 plus custom gen4 placement value; no speed rebenchmark or teacher substitution will be used to claim reuse.
