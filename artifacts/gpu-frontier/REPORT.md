# Suspended-frontier GPU generation: finite result

Adoption gate: **FAIL**. No new training or stronger model is claimed.

The prior synchronous RPC failure at fa944981 remains unchanged. Main, Web and Frozen gen3 are untouched.

| condition | work wall s | init s | games/s | nodes | NN policy/value |
|---|---:|---:|---:|---:|---:|
| cpu-1 | 96.543 | 5.165 | 1.989 | 4786820 | 1688801/2818142 |
| parallel-1 | 199.166 | 0.925 | 0.964 | 4787780 | 1689251/2818641 |
| parallel-2 | 196.607 | 0.927 | 0.977 | 4787780 | 1689251/2818641 |
| cpu-2 | 96.369 | 5.221 | 1.992 | 4786820 | 1688801/2818142 |

Paired speed ratios including initialization: [0.5083071422795082, 0.5142873514241116]; excluding initialization: [0.48473476357764417, 0.4901574557503377]. Threshold was >=1.15 in both repetitions.

## Architecture and cost
Eight CPU processes each hold eight suspended game/search generators, 64 games per seed. Seeds are processed in separate cohorts; one GPU model handles each ready wave. Recursive alpha-beta windows, TT EXACT/LOWER/UPPER, caches and RNG remain per search. Siblings remain sequential, with no speculative extra expansion. Bulk pipes replace individual node RPCs; they still serialize arrays, so this is not zero-copy. GPU batches policy and leaf value separately.

Repeat1: actual mean batch 25.15, max 64, allocated peak 92.66 MiB; total process CPU 461.18s. Server frontier wait/receive 96.84s, pack+H2D 23.30s, forward+sync 58.01s, D2H 5.25s, reply send 13.65s. Sampled GPU utilization mean 10.19%, peak physical VRAM 1274 MiB.
Repeat2: actual mean batch 25.15, max 64, allocated peak 92.66 MiB; total process CPU 457.74s. Server frontier wait/receive 96.58s, pack+H2D 23.05s, forward+sync 56.93s, D2H 5.14s, reply send 12.82s. Sampled GPU utilization mean 10.24%, peak physical VRAM 1258 MiB.

Forward wall includes CPU dispatch and synchronization, not pure CUDA kernel execution. Frontier wait includes worker CPU advancement, pipe receive and cohort tail/barrier waiting; these are not individually isolated. CPU seconds sum concurrent processes and are not wall time. GPU utilization is 1Hz sampled, not an exclusivity claim.

## Correctness and numerical limits
All 768 final-bank games independently coordinate-replayed and scored. Paired exact game trajectories: 382/384; identical-state move agreement 0.99980956; maximum same-state root-value error 1.1920929e-06. See audit.json for divergences and per-run training/saved-state overlaps. All nonpass moves completed depth6; no truncated teacher labels.
CPU singleton generator tests require exact values/actions, RNG and node/cutoff/cache counts. CUDA tolerance was absolute2e-5 and >=99% selected moves, not bit equality. Tests separately cover mixed-stage batch logits/top4/value, pass/terminal, complete-stage timeout fallback and pruned/unpruned trees.

## Timing reconciliation
The historical3072 games/24.98min is ~0.488s/game; old24 games/12.55s is ~0.523s/game. The apparent discrepancy mostly reflects different total game counts. They also differ in timing scope, start distributions, warmup, and historical500ms versus fixed-complete6ply. They are not used as paired speed evidence. Current CPU/GPU comparisons use exactly the same new192-game bank, fixed complete tree definition, CPU8 native, two repetitions, and separate initialization.

## Decision and limits
The fixed throughput gate failed. No24576-game expansion, additional training, checkpoint selection or strength evaluation was run. This is a completed finite engineering comparison, not completion of the user’s stronger-AI goal. The previous3072-game A remains unchanged.
Remaining costs include CPU encode/native observations/state copies and Python search advancement, small policy/value-separated GPU calls, and bulk IPC/barrier tails. The current bulk scheduler has a wave barrier: it receives one frontier from every active cohort before launching GPU work, so CPU advancement and GPU evaluation do not overlap across waves. This limitation is measured together with IPC and straggler time, not isolated causally. A possible next bounded hypothesis is native CPU state/feature packing plus shared-buffer asynchronous cohorts; it has not been implemented or demonstrated here. No worker/window sweep or teacher-depth reduction was used to rescue results.

## Restart and provenance
Atomic complete-game gzip records include task, prefix, initial RNG, all moves/scores and search stats. Restart skips task-matching completed records and regenerates interrupted games from their saved RNG; suspended stacks/partial WDL are not checkpointed. A different CUDA batch shape after restart can change floating ties, so semantic/numerical reproducibility must not be advertised as universal bit reproducibility. See resume-check.json and batch-numerics.json.
The new bank excludes saved historical starts/states and the referenced original training index. Old24 is development only. No bank here is a playing-strength confirmation. Historical giant indexes/data are referenced rather than duplicated.

Reproduction: README.md, PLAN.md, scripts/run_frontier_comparison.py, audit_frontier.py and plot_frontier.py. throughput.png shows throughput and batching, not a learning curve.

## Final validation
124 tests passed after correcting a test-only forced-pass fixture; the initial failure log is retained. A history-free source archive at d39f0ee loaded all three frozen gen3 exports and prior3072A exports, completed CPU games and passed3 Python-fallback scheduler tests. CUDA was hidden; the installed torch build was still2.7.1+cu128, so this is a CPU-device check, not a separate CPU-only-wheel installation. No model was reserialized.
Measured initialization includes model/process setup and warmup after Python imports; outer-process telemetry is only1Hz. Formal work plus measured initialization totals about10minutes. Resume checks passed for191 unchanged game files and one regenerated game; stack-level checkpoints are not provided.

日本語要約: 64探索を中断・再開し同一モデルのfrontierをまとめる実装は完了。実batch平均25.15まで改善したが、新192局×2反復でCPU8約96秒に対しGPU約197〜199秒、速度条件未達のため不採用。追加24576局生成・学習・棋力評価は実施せず、ユーザーの強化目的は未達。CPU探索進行・IPC・全cohort待ちとGPUの小規模呼出が残る。前段の同期RPC失敗と既存候補は保持。
