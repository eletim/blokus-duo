# Ready-frontier fixed-teacher experiment
Preserve fa944981 and synchronous RPC failure unchanged. No main/Web/Frozen changes.
Use recursive generators with per-search TT, alpha/beta, policy/value caches and
sequential sibling order. Suspend at NN calls, not at worker RPC. One process holds
64 independent games of one seed at a time; CUDA batches their ready requests by
policy/value kind. No speculative siblings. CPU8 native is the reference.

Predeclared tolerances: absolute logits/value/root errors <=2e-5, >=99% same-state
selected moves; report ties and pruning differences, never claim bit equivalence
on CUDA. CPU singleton transport must exactly match reference tree/actions/RNG.
Fixed width4 iterative depths2/4/6, no time truncation in throughput; terminal/pass
semantics and only complete game labels. Resume at game boundaries by atomic files
and per-game saved initial RNG; interrupted games restart, not partial labels.

Development: old24 bank for implementation probes only. Final bank:192 new games,
64 per frozen model, independent RNG, starts excluded from known saved states and
original PPO index. Two repeats CPU/GPU/GPU/CPU, initialization separately and
included, no heavy concurrent job. Max stage1 compute budget60min. One profiling-led
implementation correction allowed before final bank timings; no batch/window sweep.
Gate: both repeated end-to-end throughput ratios >=1.15 versus CPU8, complete depth6,
legality/score replay, numerical tolerance and model SHA checks. If failed, no large
training. If passed, commit separate finite data/training/evaluation plan before
strength outcomes. Record batch sizes, CPU/transfer/forward/wait, memory, utilization,
nodes/NN requests, games/s and stage completion. New bank is not strength evidence.
