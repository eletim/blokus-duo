# 4ch PPO利用候補（gen2置換は非推奨）

3seedの新規4ch学習。heuristic priorを保持。全seedで65536gamesまで学習し、新validationだけで4096/2048/4096iterationを選択。seed1751402の配布モデルは32768games時点。学習前/heuristic比point80.56%だが、公開gen2比48.00%（95% CI43.75–52.34%）、得点差-0.39（-1.17,+0.41）。96prefix内で3seedと先後をまとめてbootstrap。gen2改善未確認。checkpoint/seedのholdout後再選択なし。

公開gen2/2048・既定値は維持。全モデルは4ch形式識別付きで、履歴sidecar不要。CPU/CUDAで利用可能。代表例seed1751401は成績によるseed選別ではない。

```bash
python -m blokus.ppo_play --checkpoint artifacts/ppo-four-gpu-models/seed-1751401.pt --device cpu --backend auto --human-seat 0
# seed-1751402.pt / seed-1751403.ptも明示指定できる
```

探索なし。対gen2もネット単体かつnative観測backendで測定。4chの推論は約0.521ms/手、gen2約0.521–0.537ms/手（CUDA、今回の対局分布、初期化除外）。一般的最強や4ch化による棋力向上は主張しない。研究branchの artifacts/ppo-four-gpu/REPORT.md に全記録・制約。
