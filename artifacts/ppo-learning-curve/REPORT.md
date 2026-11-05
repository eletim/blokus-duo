# 終局PPO 学習量曲線

main推奨は変更しない。盤面・残駒・正確な候補配置を入力する全合法手のネット単体。固定heuristic priorとPPO、終局勝敗報酬、現在/過去/heuristic相手割合は前回と同じ。探索教師や追加探索は用いない。

## 条件と選択

3seedとも最新512から継続。前回selected (seed1401:128、1402/1403:512) を固定比較基準とした。validation32prefixで previous .5 / heuristic .25 / oldvalue .25 のpointを最大化し、同値は得点差・早いiteration。前回selected自体も候補。holdoutを選択に使用しない。

|seed|終了iteration|選択iteration|追加wall秒|新規対局数|
|---|---:|---:|---:|---:|
|1401|2048|2048|998.5|24576|
|1402|2048|2048|1019.0|24576|
|1403|2048|2048|1005.0|24576|

## 学習量別validation / dev診断

|seed/checkpoint|対前回selected point / 得点差|対heuristic point / 得点差|対oldvalue point / 得点差|dev greedy / stochastic point|
|---|---|---|---|---|
|1401/previous|0.500 / +0.000|0.484 / -0.203|0.242 / -4.734|0.625 / 0.531|
|1401/iteration-1024|0.719 / +3.188|0.805 / +6.406|0.344 / -3.516|0.750 / 0.688|
|1401/iteration-2048|0.750 / +5.266|0.773 / +6.469|0.523 / +0.797|0.688 / 0.500|
|1402/previous|0.500 / +0.000|0.594 / -0.266|0.328 / -2.844|0.531 / 0.406|
|1402/iteration-1024|0.578 / +1.125|0.742 / +4.281|0.438 / -2.922|0.750 / 0.594|
|1402/iteration-2048|0.781 / +6.391|0.750 / +6.484|0.703 / +4.484|0.875 / 0.875|
|1403/previous|0.500 / +0.000|0.500 / -0.391|0.398 / -3.188|0.594 / 0.469|
|1403/iteration-1024|0.781 / +6.484|0.695 / +2.875|0.414 / -3.219|0.688 / 0.500|
|1403/iteration-2048|0.844 / +8.766|0.781 / +7.156|0.609 / +3.391|0.812 / 0.656|

## 独立holdout96prefix・先後交換

pointは勝ち1/引分0.5/負け0。同一prefix内で2座席と3seedを平均し、96prefixをbootstrap。更新seed3つの結果は別途JSONに保存し、独立576標本と扱わない。

|対戦相手|point [95% CI]|平均得点差 [95% CI]|
|---|---|---|
|previous|0.813 [0.785, 0.843]|+7.203 [+6.562, +7.856]|
|heuristic|0.791 [0.759, 0.825]|+6.205 [+5.439, +7.005]|
|oldvalue|0.613 [0.575, 0.649]|+2.104 [+1.272, +2.891]|

## 分離と検証

全訓練対局を独立座標合法性checkerで再生し、相手手番・prefix・終局を含む全訪問hashを生成。新評価開始状態はその集合と過去評価prefix/開始状態から除外。評価前に固定し成績で選び直していない。空盤面等の共通prefix途中状態は訓練側の除外集合には含むが、評価の意思決定局面とは区別する。

評価後の実訪問状態照合: {"dev": {"games": 320, "unique_decision_states": 9237, "training_actual_visited_overlap": 0, "adopted_start_overlap": 0}, "validation": {"games": 1728, "unique_decision_states": 45083, "training_actual_visited_overlap": 0, "adopted_start_overlap": 0}, "holdout": {"games": 1728, "unique_decision_states": 50245, "training_actual_visited_overlap": 0, "adopted_start_overlap": 0}}

CUDA別process再開は512→518連続6更新iterationと、512→514保存＋514→518追加4iteration。516で過去policyを交換。model/past/optimizer/全RNG/対局履歴bit一致。

## 実時間と限界

追加生成＋更新＋IO wall合計 3022.6秒。trainログに相手別成績、CPU生成、GPU推論同期、更新時間を保存。ネット全合法手生成・GPU同期を含む実msを各対局JSONへ保存。旧value/2plyは6ms設定の探索であり、ネット単体と同じ思考時間条件ではない。

dev8prefixのgreedy対確率的評価は診断専用。主評価は一貫してgreedy。前回selectedへの信頼区間とseed差を優先し、heuristicへの改善のみで採用しない。今後は量を無制限に増やさず、凍結相手を複数世代保持するpool等を別実験として検討する。

## 再現

```bash
for seed in 1401 1402 1403; do
  .venv/bin/python -m blokus.ppo_scale --seed "$seed" --resume "artifacts/terminal-ppo/extended/seed-$seed/iteration-512.pt" --output "/tmp/ppo-repro/seed-$seed" --iterations 2048 --cap 1200
done
.venv/bin/python scripts/check_ppo_scale_resume.py
.venv/bin/python -m pytest -q
# 記録済みfixtures/resultsを保持して再監査
.venv/bin/python -m blokus.ppo_curve_eval audit_train
.venv/bin/python -m blokus.ppo_curve_eval audit_eval
```

学習checkpointはhistory sidecar SHAを参照。パスと選択hashはselection.json。再評価は既存結果を上書きせず別出力先を使用する。

## 結論と実測内訳

3seedすべてで前回selectedへの改善が新holdoutでも確認された。1024から2048でvalidationの前回selected対戦成績も全seedで上昇した。学習量による改善を支持する結果であり、探索追加や報酬変更によるものではない。ただしこの固定相手・prefix分布に対する結果で、万能な棋力向上や同思考時間での優越を主張しない。mainへの統合・推奨変更は行っていない。

|相手|勝/敗/分（3seed合計）|ネット実平均ms|相手実平均ms|
|---|---|---:|---:|
|previous|462/101/13|1.327|1.442|
|heuristic|445/110/21|1.207|0.705|
|oldvalue|343/213/20|1.215|4.643|

旧2plyに対するseed1401のpoint 95%CIは[0.495, 0.628]で0.5を含む。一方、得点差CIは[+0.031,+2.615]。seed1402/1403はpoint/得点差とも正の区間。この3モデルに条件づけたprefix不確実性であり、更新seed母集団への狭いCIではない。

追加213158 minibatch更新。生成2336.1秒（うちCPU合法手生成・pack 2022.0秒、GPU推論同期区間258.3秒）、PPO更新654.8秒。GPU推論同期区間には転送等も含み、純粋kernel時間ではない。その他の生成処理・loading・IO・待ちを含め総3022.6秒。CPU生成が主な律速のままで、今回は挙動を変える並列化をしていない。

全98tests PASS、CUDA別process再開bit一致、全98304訓練対局と3776評価対局の再生/算術監査PASS。全評価実訪問状態の訓練重複0。AI対AI CLIも終局まで完走。これは工房実装の自己監査であり、秘書による独立実装監査ではない。

dev8prefixではgreedyが確率的評価以上のpointとなった（10モデル中9つで上、1つ同点）。小標本の診断であり、評価法を選び直す材料にはしていない。entropyは最後64iteration平均で1.12〜1.25、value MSEは0.60〜0.68。最終結果への判断は訓練MSEではなく分離済み対戦を用いた。

次段階候補: 同条件の有限な4096iteration延長で曲線が続くかを検証する、または複数世代の凍結policy/より強い相手を含むpoolを別条件として比較する。両変更を同時に行わず、今回のselected2048を固定基準として保全する。100ms教師との比較は今回未実施。

研究モデル対戦例（seed1401は記載例でありholdoutで最良seedを再選択していない）:

```bash
.venv/bin/python -m blokus.terminal_play --model artifacts/ppo-learning-curve/seed-1401/iteration-2048.pt --opponent heuristic
.venv/bin/python -m blokus.terminal_play --model artifacts/ppo-learning-curve/seed-1401/iteration-2048.pt --human-seat 1
```

補足診断: seed1401の再開元512は前回selected128と異なるため、選択固定後に同じdev8prefixでgreedy/確率的比較だけを追記した（point .9375/.5000）。学習量の増加とdev成績は単調ではなく、小標本の揺らぎが大きい。これによるcheckpoint再選択はしていない。seed1401について今回の主比較はselected128との比較であり、latest512とのfresh96prefix直接比較は未実施。seed1402/1403は前回selected自体が512である。

固定済みfixtureで実対戦を再実行するCLI（出力が存在する場合は上書きせず拒否）:

```bash
.venv/bin/python -m blokus.ppo_curve_eval match --left artifacts/ppo-learning-curve/seed-1401/iteration-2048.pt --right oldvalue --fixture artifacts/ppo-learning-curve/holdout-prefixes.json --output /tmp/ppo-curve-replay.json
```
