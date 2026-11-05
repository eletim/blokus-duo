# PPO固定2048再確認・main提供・4096学習量曲線

## 固定2048の再確認

3seed/重み/設定を固定。新seed1570000の96prefix・先後交換。モデル再選択なし。100ms教師はpriority/予算100ms/追加reserve0、試合ごとにcost/refine見積りをcold reset。以前のwarm条件を改変した再計算ではなく、新しい独立標本。工房側の同実装検証であり秘書の独立実装監査とは呼ばない。

|相手|point [95% CI]|得点差 [95% CI]|PPO/相手の実平均ms|
|---|---|---|---|
|oldvalue|0.561 [0.5182291666666669, 0.6015625]|+0.696 [-0.1946180555555556, 1.590277777777777]|1.218/4.690|
|teacher100|0.450 [0.4114583333333332, 0.4887152777777777]|-2.026 [-3.086805555555555, -0.9878038194444444]|1.346/70.388|

旧6msへの平均得点差の正の再確認は不確実。100ms教師には負けており、一般的最強とは扱わない。seed1401の2048対再開元512は147勝36敗9分、+7.536 [6.135,8.938]。これはselected128との比較ではなく512への直接比較。

## main提供

main d8b9803（推論cherry-pick 6fbd3e9）。全合法手のPPOを追加選択肢として提供し、旧低予算AIと100ms教師/既存デフォルトを維持。既定例seed1401は評価前固定。exportは各約276KiB、weights_onlyロード、履歴/optimizer/RNG不要。3seed×CPU/CUDA×26局面で元モデルとの入力/logits/value/greedy bit一致。新cloneでCPU専用2.7.1+cpuの3seed・人間先後・旧6ms/教師対戦、CUDA対戦を確認。main57tests PASS。

## 継続条件と選択

各seed2048のoptimizer/past/RNGを継承。loss、terminal勝敗報酬、学習率、16ゲーム同期生成、current/past/heuristic比率、4iterationごとのpast交換を変更せず3072/4096を保存。再確認開始状態をprefix途中に含む生成だけは棄却し、件数を記録。損失/相手pool変更とは区別する。学習履歴は旧2048と追加分をSHA付きchain参照し重複保存しない。

新validation seed1590000/32prefixで2048を候補に残し previous .5 + heuristic .25 + oldvalue .25 のpoint、同値なら同重み得点差・早いiterationで選択。選択後fresh seed1600000/96prefix、先後交換で対2048/旧6msを評価。訓練全訪問状態と過去評価prefix/開始状態を事前除外し、事後にも実訪問を監査。

|seed|終了iteration|選択iteration|追加wall秒|追加games|
|---|---:|---:|---:|---:|
|1401|4096|3072|1327.7|32768|
|1402|4096|4096|1334.8|32768|
|1403|4096|4096|1318.2|32768|

### 学習量別validation

|seed/checkpoint|対2048 point / 得点差|対heuristic point / 得点差|対旧6ms point / 得点差|
|---|---|---|---|
|1401/previous|0.500 / +0.000|0.805 / +5.812|0.477 / -0.547|
|1401/iteration-3072|0.570 / +0.469|0.875 / +7.344|0.477 / -0.281|
|1401/iteration-4096|0.539 / +1.109|0.789 / +6.000|0.539 / +0.609|
|1402/previous|0.500 / +0.000|0.859 / +7.906|0.656 / +3.359|
|1402/iteration-3072|0.539 / +0.344|0.828 / +8.078|0.594 / +2.484|
|1402/iteration-4096|0.633 / +2.734|0.820 / +7.172|0.492 / +1.281|
|1403/previous|0.500 / +0.000|0.844 / +7.391|0.531 / -1.031|
|1403/iteration-3072|0.539 / +1.688|0.766 / +5.594|0.523 / +0.188|
|1403/iteration-4096|0.578 / +3.875|0.883 / +9.125|0.656 / +3.922|

## 選択固定後holdout

2座席と3seedをprefix内平均し96prefix bootstrap。これは固定した3モデルに条件づけた区間で、更新seed母集団へのCIではない。ネット単体と旧6ms探索の実時間は異なる。

|相手|勝/敗/分|point [95% CI]|得点差 [95% CI]|
|---|---|---|---|
|previous|355/191/30|0.642 [0.6024305555555557, 0.6805555555555558]|+3.562 [2.7430121527777778, 4.376866319444448]|
|oldvalue|335/210/31|0.609 [0.5729166666666665, 0.6449869791666669]|+2.562 [1.8263454861111108, 3.269140625000001]|

## 検証・計算量

評価実訪問の訓練重複: {"validation": {"games": 1728, "unique_decision_states": 47071, "training_actual_visited_overlap": 0, "adopted_start_overlap": 0}, "holdout": {"games": 1152, "unique_decision_states": 33934, "training_actual_visited_overlap": 0, "adopted_start_overlap": 0}}

CUDA別process再開は2048→2054連続6iterationと2048→2050保存＋追加4。2052のpast交換を含むmodel/past/optimizer/全RNG/履歴bit一致。報酬符号・合法性・終局得点・選択最大criterion・checkpoint SHA・試合算術を監査。

追加wall計3980.7秒、追加98304games、267369minibatch更新。CPU生成/GPU推論同期/更新の内訳はcompute.json。曲線はlearning-curves.png。同期区間は純kernel時間ではない。

## 再現

```bash
for seed in 1401 1402 1403; do
  .venv/bin/python -m blokus.ppo_scale --seed "$seed" --resume "artifacts/ppo-learning-curve/seed-$seed/iteration-2048.pt" --output "/tmp/ppo4096-repro/seed-$seed" --iterations 4096 --save-steps 3072 4096 --cap 1800 --forbidden-prefixes artifacts/ppo-4096/confirmation-prefixes.json
done
.venv/bin/python scripts/check_ppo_4096_resume.py
.venv/bin/python -m pytest -q
.venv/bin/python -m blokus.ppo_extend audit_eval
.venv/bin/python -m blokus.ppo_curve_eval match --left artifacts/ppo-4096/seed-1401/iteration-3072.pt --right oldvalue --fixture artifacts/ppo-4096/holdout-prefixes.json --output /tmp/ppo4096-repeat.json
```

mainの利用者はPPO.mdのCLIを使用。新継続モデルは研究branchのみ。今回の結果で2048のmain既定seedを選び直さず、追加学習もこの有限runで終了。

## 判断と制約

追加学習後のvalidation選択済みモデルは、固定2048と旧6ms探索の両方に、新holdoutで得点差・pointの改善を示した。ただしseed1401は4096ではなく3072が選択されており、全seedで更新数に単調な改善があるとはいえない。100ms教師との比較は固定2048についてのみ実施し、新選択モデルが教師を超えるかは未測定。mainには2048のみを追加し、新モデルの採用判断は秘書へ返す。

主holdoutの実平均時間は新モデル/固定2048が1.444/1.263ms、新モデル/旧6ms探索が1.270/4.602ms。合法手生成と同期を含み、同時間探索比較ではない。追加生成CPU処理が依然として主費用。終盤報酬のみのvalue MSEは末尾64iterationで約0.62〜0.69、entropy約0.80〜0.89で、非有限値や一律の方策崩壊は見られなかった。

次段階は今回の選択済みモデルを固定した再確認または100ms教師との比較を先に行い、追加学習量・相手poolの変更は別実験として扱う。今回の結果だけで無制限の継続学習は行わない。

開始から有限pipeline（再確認・main統合・継続・評価監査）完了まで94.0分。生成＋更新は66.3分。最終文書整理・push時間はこのpipeline値に含まない。再確認開始状態の混入防止で棄却したprefix件数はtraining-audit.jsonにseed別記録。全訓練ゲームの終局報酬再生と、新評価の実訪問状態非重複を確認した。
