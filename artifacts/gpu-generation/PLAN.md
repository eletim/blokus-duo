# GPU generation stage 1 (before measurements)

Preserve all previous studies, main, Web and Frozen weights. No training yet.
CPU rules/native observations stay in eight independent processes. One central CUDA
inference service groups requests by frozen model and policy/value input type;
workers retain the existing deterministic depth2/4/6 alpha-beta algorithm, width4.
Use a generous 60-second safety deadline; reject every nonpass move not completed
at depth6. This is fixed-work testing, not comparison against prior 500ms timings.

Fixed bank: 24 new random-eight-ply prefixes, eight per frozen gen3 seed. RNG per
(seed, game) independent of scheduling. CPU8 and GPU8 run the identical bank twice
in alternating order CPU/GPU/GPU/CPU. Initialization/warmup reported separately.
Atomic per-game completion and deterministic replay from its initial RNG provide
restart at game boundaries; interrupted games are discarded, never WDL labels.
No batching or depth rescue sweep after results. Batch cap32, wait up to1ms.

Before games, compare identical observations (opening/middle/end/pass) on CPU and
CUDA: absolute value/logit tolerance2e-5, finite logits only, report maximum errors,
full ranking/top4 and selected-move agreement. Floating order is not bit equivalence;
near ties and changed pruning may alter trees. Completed fixed-tree root values
must be within2e-5 on saved states; report any move disagreement, never conceal it.
Game-level speed gate: both paired repeats >=1.15x throughput, all complete depths,
independent legality/score replay, and >=99% move agreement on identical probes.
If gate fails, stop before data expansion/training; retain implementation and report
limitations. Success permits a separately committed finite stage2 plan before
strength outcomes. No claims of strength from throughput.

Record wall, process CPU, request waiting, pack, CUDA event kernel/transfer time,
batch histogram, padding, peak allocated VRAM, sampled GPU utilization, nodes and
completed games/sec. Same CPU cores/worker count. No concurrent training/evaluation.
Development benchmark games are not training or strength confirmation samples.
Exclude known prior train/dev/evaluation starts using saved exclusions/index; audit
all generated visited states and mark any overlaps; do not turn them into labels.
