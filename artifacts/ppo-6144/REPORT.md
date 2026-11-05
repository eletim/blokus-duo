# PPO4096→6144 有限継続

mainはgen2のまま。3seed最新4096のmodel/optimizer/past/RNGから、同じloss/終局勝敗報酬/モデル/LR .0003/16ゲーム/current .5,past .25,heuristic .25/4iterationごとのpast交換で継続。seed1401のgen2は3072であり、再開元4096とは別モデル。これは同条件の有限追加学習であり、探索やpool変更は行っていない。

## 計算と学習量

|seed|終了iteration|選択iteration|生成更新保存wall秒|追加games|
|---|---:|---:|---:|---:|
|1401|6144|5120|1314.0|32768|
|1402|6144|5120|1281.3|32768|
|1403|6144|6144|1268.4|32768|

CUDA別process再開smoke込みの保守的wall合計 3904.9秒（上限5400秒）。内訳compute.json、学習曲線learning-curves.png。更新minibatch計243096、追加98304局。

## 分離と選択

既存の全評価prefix 2434件を凍結し、その開始状態を途中に含む新生成prefixを棄却。個別ゲームのprefix乱数だけを進め、他ゲームのprefix乱数は変えない。従来のprefix列と完全同一ではないがloss/pool等は固定。旧圧縮履歴と旧全訪問indexはSHA参照、追加データだけ保存。全訓練履歴を再生して報酬符号・合法性・得点を監査。

{"pass_": true, "rows": [{"seed": 1401, "new_games": 32768, "prefix_rejections": 84, "prefix_start_hits": 0, "incidental_actual_visited_start_overlap": 0}, {"seed": 1402, "new_games": 32768, "prefix_rejections": 75, "prefix_start_hits": 0, "incidental_actual_visited_start_overlap": 0}, {"seed": 1403, "new_games": 32768, "prefix_rejections": 63, "prefix_start_hits": 0, "incidental_actual_visited_start_overlap": 0}], "note": "Historical training may precede old evaluations. Exclusion is audited for new games only. Incidental rollout visits are recorded separately from prefix rejection."}

新validation seed1650000/32prefixでgen2 .5、旧6ms .25、教師100ms .25のpointを選択criterionにし、同値時は同重み得点差、さらに早いiteration。gen2・latest4096も候補に残す。seed1401のlatest4096直接比較は別記しcriterionに加えない。選択固定後、fresh seed1660000/96prefixでgen2と教師へ先後交換。seed1401のlatest4096補足は結果を見る前から予定し、全モデルの再選択には使わない。

|seed/候補|対gen2 point/得点差|旧6ms|教師100ms|対latest4096（1401のみ）|
|---|---|---|---|---|
|1401/gen2|0.500/+0.000|0.641/+2.469|0.422/-2.047|0.602/+0.547|
|1401/latest4096|0.398/-0.547|0.586/+1.406|0.445/-3.656|0.500/+0.000|
|1401/iteration-5120|0.547/+1.812|0.555/+1.453|0.523/+0.984|0.594/+0.531|
|1401/iteration-6144|0.562/+0.750|0.586/+1.078|0.430/-0.234|0.578/+1.109|
|1402/gen2|0.500/+0.000|0.656/+3.984|0.484/+0.172|gen2と同一|
|1402/iteration-5120|0.547/+1.500|0.680/+3.875|0.539/+1.578|gen2と同一|
|1402/iteration-6144|0.492/+1.125|0.742/+5.609|0.484/+0.859|gen2と同一|
|1403/gen2|0.500/+0.000|0.656/+4.922|0.555/+1.531|gen2と同一|
|1403/iteration-5120|0.469/-0.625|0.633/+3.797|0.555/+2.562|gen2と同一|
|1403/iteration-6144|0.562/+1.297|0.820/+7.312|0.547/+2.109|gen2と同一|

## 選択固定後fresh holdout

CIはprefix単位bootstrap、2座席・3seedをprefix内平均する。固定した選択済み3モデルに条件づけた区間で、更新seed母集団へのCIではない。ネット単体と100ms教師は思考時間が異なる。教師のcost/refine推定を試合ごとcoldにし、生成・学習・監査と時間対戦を重ねていない。

|相手|point [95% CI]|得点差 [95% CI]|選択モデル/相手 実ms|
|---|---|---|---|
|gen2|0.5148 [0.48174913194444424, 0.5486111111111109]|+0.6979 [-0.04166666666666655, 1.4253906250000004]|1.353/1.322|
|teacher100|0.5955 [0.5616319444444445, 0.6302083333333335]|+2.0833 [1.342013888888889, 2.8768229166666672]|1.404/68.879|

|seed/相手|勝/敗/分|point [95% CI]|得点差 [95% CI]|
|---|---|---|---|
|seed-1401/holdout-gen2.json|93/92/7|0.5026 [0.4348958333333333, 0.5677734375000003]|+0.6979 [-0.5834635416666667, 1.953125]|
|seed-1401/holdout-latest4096.json|100/84/8|0.5417 [0.4739583333333333, 0.609375]|-0.0104 [-1.3697916666666667, 1.3333333333333333]|
|seed-1401/holdout-teacher100.json|102/82/8|0.5521 [0.4921875, 0.6145833333333334]|+0.9583 [-0.526171875, 2.4376302083333337]|
|seed-1402/holdout-gen2.json|92/94/6|0.4948 [0.4296875, 0.5625]|+0.6875 [-0.6147135416666667, 1.9739583333333333]|
|seed-1402/holdout-teacher100.json|106/80/6|0.5677 [0.4921875, 0.6380208333333334]|+1.3385 [-0.026171874999999956, 2.7135416666666665]|
|seed-1403/holdout-gen2.json|102/84/6|0.5469 [0.484375, 0.6119791666666666]|+0.7083 [-0.4010416666666667, 1.8179687500000008]|
|seed-1403/holdout-teacher100.json|125/61/6|0.6667 [0.6015625, 0.7317708333333334]|+3.9531 [2.504947916666667, 5.448046875000001]|

## 検証

{"validation": {"games": 2176, "unique_actual_states": 56894, "training_actual_overlap": 0, "adopted_start_overlap": 0}, "holdout": {"games": 1344, "unique_actual_states": 40266, "training_actual_overlap": 0, "adopted_start_overlap": 0}}

CUDA別processで4096→4102連続6iterationと4096→4098保存＋追加4を比較。4100の過去policy交換を含めmodel/past/optimizer/Python/NumPy/Torch/CUDA RNG/履歴bit一致。全テスト、訓練報酬/終局/合法性、checkpoint SHA、criterion最大値/算術、評価開始状態/実訪問状態の訓練分離を監査。初期空盤面やprefix再生途中は評価decisionと区別している。

## 再現

```bash
for seed in 1401 1402 1403; do
  .venv/bin/python -m blokus.ppo_scale --seed "$seed" --resume "artifacts/ppo-4096/seed-$seed/iteration-4096.pt" --output "/tmp/ppo6144-repeat/seed-$seed" --iterations 6144 --save-steps 5120 6144 --cap 1760 --forbidden-prefixes artifacts/ppo-6144/forbidden-prefixes.json
done
.venv/bin/python scripts/check_ppo_6144_resume.py
.venv/bin/python -m blokus.ppo_6144_eval audit_eval
.venv/bin/python -m blokus.ppo_6144_eval match --left artifacts/ppo-6144/seed-1401/iteration-5120.pt --right teacher100 --fixture artifacts/ppo-6144/holdout-prefixes.json --output /tmp/ppo6144-teacher-repeat.json
.venv/bin/python -m blokus.terminal_play --model artifacts/ppo-6144/seed-1401/iteration-5120.pt --opponent artifacts/ppo-4096/seed-1401/iteration-3072.pt
```

保存済みprefixを成績で再選択しない。既存run出力の上書きは拒否する。教師の時間制御探索はOS/実時間に依存するので試合結果のbit再現は保証しない。今回の有限runで停止しmain推奨を変更しない。

## 判断と次案

対gen2はpoint 51.48% [48.17,54.86]、得点差+0.698 [-0.042,1.425]で、今回の追加学習による安定した改善は未確認。各seedの得点差もCIが0を含む。対100ms教師は固定3モデル平均でpoint59.55% [56.16,63.02]、得点差+2.083 [1.342,2.877]。seed1403の効果が大きく、1401/1402単独ではCIが同等を含む。教師との勝利をgen2からの改善の証拠へ置き換えず、一般的最強とも呼ばない。seedの再選択なし、main gen2を維持する。

1401の5120対再開元4096は100勝84敗8分、point54.17% [47.40,60.94]、得点差−0.010 [-1.370,1.333]。対gen2(3072)のpoint50.26%とも区別し、再開元からの改善も確定できない。

今回の範囲で同条件の追加量には頭打ちの兆候がある。次案は相手poolの過去policy枠25%だけを「直近past＋凍結gen2/2048」に分散する対照実験。現行の直近past単独poolと同生成更新wallで比較し、loss・報酬・モデルを固定する。古い世代への頑健性を狙う仮説であり、この敗勝結果だけから原因を断定しない。自動延長や新学習は実施していない。

## 再起動復旧と監査経緯

初回の追加分離再生でpassの独立全配置checkerを重複実行し遅延したため、その監査だけ停止。先行して完了した全履歴合法性・報酬監査とcheckpoint/history SHAを確認後、分離用の再生だけ重複検査を省いて再開した。旧pipeline.logを保存し、学習・対戦結果は再実行していない。

complete.jsonの完了時刻とstarted.jsonの差は6872.0秒（114.5分）。生成・更新・保存の3process合計3868.25秒、再開検証36.62秒を含む保守的計算wall3904.86秒（65.08分）。計測は再起動前のRTX5090環境。休止期間は計算時間に含めない。教師100ms超過率0%、教師最大95.31ms、ネットは固定deadlineのないgreedy推論で、同時間対照ではない。

2026-11-06明示再開後、保存済み6checkpointのmodel/pastロード、optimizer/RNG存在、全15履歴参照のSHA、選択hash、CPU/CUDA forwardの有限性と出力一致（rtol/atol 1e-4）を確認。全109テストPASS（9.04秒）。元evaluation-audit.jsonをpre-reboot-evaluation-audit.jsonに保全し、2176 validation局＋1344 holdout局の再生・算術・SHA・分離監査を再実行。実対局は再実行せず、監査JSONは再起動前と一致。全訓練訪問状態との採用開始/実訪問重複0。別process CUDA再開は既存の4096→4102連続6と4096→4098＋追加4のbit一致記録を保持し、学習を繰り返さない。

復旧検証: `.venv/bin/python scripts/check_ppo_6144_recovery.py`。全テスト: `.venv/bin/python -m pytest -q`。最終報告再生成: `.venv/bin/python scripts/report_ppo_6144.py`。履歴を参照する研究checkpointは研究branchの既存成果と併せて利用する。mainへの新モデル提供は今回行わない。
