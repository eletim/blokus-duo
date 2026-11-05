# 複数世代pool A/B 対照実験

main gen2維持。各seed最新6144の同一model/optimizer/past/RNGから分岐し、同じloss/報酬/モデル/LR/batch16/2epochs/early-KLで有限継続。A current50/past25/heuristic25。B current50/heuristic25/past・同seedgen2・同seed2048を各約8.33%。3iterationで過去枠比率均等、6iterationで先後均等、端数は以下の実数を参照。更新はcurrent自己対戦の双方とその他対局のlearner席のみ。

最新6144は前回selected5120/5120/6144と区別する。固定gen2は3072/4096/4096、固定2048は各seed2048。全SHAはlock.jsonとpool-B.json。既存の成績をAの代用にせずAも新規実行した。

## 計算量・割合

|条件/seed|追加局|PPO呼出|minibatch更新|wall秒|CPU生成秒|GPU推論同期秒|更新秒|
|---|---:|---:|---:|---:|---:|---:|---:|
|A/1401|16384|1024|40282|647.0|437.6|57.9|123.2|
|A/1402|16384|1024|38325|629.7|428.2|56.7|116.9|
|A/1403|16384|1024|35212|635.7|443.0|56.8|107.8|
|B/1401|16384|1024|38515|672.3|450.2|74.5|119.2|
|B/1402|16384|1024|38189|666.6|448.5|73.5|116.3|
|B/1403|16384|1024|36252|644.5|431.7|73.5|110.9|

6run合計wall 3895.8秒（生成＋更新＋保存、上限5400秒）。追加ゲーム数とPPO呼出数を揃え、実minibatch数や学習state数の差を隠さない。early-KLを変えて厳密な等勾配step数にした実験ではない。

|条件/seed|対局割合|学習state割合|
|---|---|---|
|A/1401|{"current": 0.5, "heuristic": 0.25, "past": 0.25}|{"current": 0.6640304403748393, "heuristic": 0.16995365549907107, "past": 0.16601590412608971}|
|A/1402|{"current": 0.5, "heuristic": 0.25, "past": 0.25}|{"current": 0.6617285561441507, "heuristic": 0.1726651183424232, "past": 0.16560632551342605}|
|A/1403|{"current": 0.5, "heuristic": 0.25, "past": 0.25}|{"current": 0.6623492906432839, "heuristic": 0.17185626585567923, "past": 0.1657944435010369}|
|B/1401|{"current": 0.5, "gen2": 0.08331298828125, "gen2048": 0.08331298828125, "heuristic": 0.25, "past": 0.0833740234375}|{"current": 0.6635774882400111, "gen2": 0.055492846964365065, "gen2048": 0.05561798039746464, "heuristic": 0.169956739584557, "past": 0.05535494481360226}|
|B/1402|{"current": 0.5, "gen2": 0.08331298828125, "gen2048": 0.08331298828125, "heuristic": 0.25, "past": 0.0833740234375}|{"current": 0.6625264393015393, "gen2": 0.055346271222203355, "gen2048": 0.05540802725499581, "heuristic": 0.17163545413842615, "past": 0.055083808082835424}|
|B/1403|{"current": 0.5, "gen2": 0.08331298828125, "gen2048": 0.08331298828125, "heuristic": 0.25, "past": 0.0833740234375}|{"current": 0.662106976093367, "gen2": 0.054980591731922324, "gen2048": 0.055514383884659436, "heuristic": 0.17210910089708176, "past": 0.05528894739296949}|

## 選択と進行gate

事前指定した共通最大保存iterationのみ採用。途中checkpointを成績で選ばない。全seedを保持。共通prefix乱数・除外集合・初期RNGを共有するが、行動変化・対局長・model別バッチ構成によりCUDA乱数消費と軌跡は変わり、完全な共通乱数対戦とは呼ばない。学習prefix一致はtraining-audit.jsonで検証。

{"A": {"1401": {"path": "artifacts/ppo-generation-pool/A/seed-1401/iteration-7168.pt", "sha256": "f800777d7d947856bb15172443f94d1c5614df5208e28291f9188f3d3c8f7e06", "iteration": 7168}, "1402": {"path": "artifacts/ppo-generation-pool/A/seed-1402/iteration-7168.pt", "sha256": "58aa2faef380708eb61d6ac2247975b3c5117317b4dd2db5836d60b0cccd340f", "iteration": 7168}, "1403": {"path": "artifacts/ppo-generation-pool/A/seed-1403/iteration-7168.pt", "sha256": "c5da8cbe7864a4f46f1f682c0c7b03aa2a27795e2ee76694bc2eaa3c8e2f0fa5", "iteration": 7168}}, "B": {"1401": {"path": "artifacts/ppo-generation-pool/B/seed-1401/iteration-7168.pt", "sha256": "5da51256c7ddf4d90b2bb1f0a57b142bce35371e8971abddc821acaadaf857f5", "iteration": 7168}, "1402": {"path": "artifacts/ppo-generation-pool/B/seed-1402/iteration-7168.pt", "sha256": "98564066a476422699b34b2a8b2f5daf16699becf823173be244dc84925f2c72", "iteration": 7168}, "1403": {"path": "artifacts/ppo-generation-pool/B/seed-1403/iteration-7168.pt", "sha256": "b740d38d57d3edc9d70d86508b9fe5c42a6c9149da9f124c3c05504fd253f249", "iteration": 7168}}}

新validation1670000/32prefixで前回selected/gen2/旧6ms/教師100msを評価。gateはB対A point>=.45かつ固定4相手の平均point B−A>=−.05、合法性/有限/分離監査PASS。これは大規模比較への進行条件であり強化成功判定ではない。

{"advance": true, "B_vs_A_point": 0.5755208333333334, "fixed_opponent_B_minus_A": -0.02734375, "gate": "BvsA>=.45 and four-opponent B-A>=-.05; common checkpoint fixed by compute, never by performance", "selection_sha256": "3640114d54c7176e51c95afd2baf866e6da7547e7f0a41861294296d833e5b9d"}

## 対戦結果

主評価は同追加経験量のB対A。新holdout1680000/96prefix、先後交換。CIはprefix単位bootstrap、2席と3seedをprefix内集約した固定モデル条件付きCIで、seed母集団の不確実性は含まない。教師100msは試合cold、時間計測は生成・学習・監査と直列。ネット単体と100ms教師は異なる実費用。

|phase/相手|point [95% CI]|得点差 [95% CI]|B/相手ms|
|---|---|---|---|
|validation-A|0.5755 [0.5130208333333334, 0.6380208333333334]|+1.0312 [-0.770963541666667, 2.8125]|1.296/1.287|
|validation-gen2|0.5391 [0.48697916666666663, 0.5885416666666669]|+0.6823 [-0.5156249999999999, 1.8489583333333333]|1.336/1.246|
|validation-teacher100|0.5365 [0.4713541666666667, 0.6015625]|+1.9792 [0.4009114583333334, 3.6458333333333326]|1.372/67.908|
|holdout-A|0.4826 [0.4383463541666665, 0.5269097222222222]|+0.1007 [-0.7657986111111114, 0.9288628472222226]|1.328/1.321|
|holdout-gen2|0.5530 [0.5164930555555555, 0.5885416666666665]|+1.6215 [0.8818576388888887, 2.3871961805555557]|1.384/1.282|
|holdout-teacher100|0.5712 [0.5269097222222225, 0.613715277777778]|+1.7587 [0.8089843750000001, 2.6407118055555565]|1.395/69.328|

|各条件/seed/相手|勝敗分|point [CI]|得点差 [CI]|
|---|---|---|---|
|A/seed-1401/validation-gen2.json|40/19/5|0.6641 [0.5703125, 0.7578125]|+4.6250 [2.34375, 6.953125]|
|A/seed-1401/validation-oldvalue.json|41/22/1|0.6484 [0.53125, 0.765625]|+5.1094 [3.171875, 6.9375]|
|A/seed-1401/validation-previous.json|42/19/3|0.6797 [0.5625, 0.7890625]|+3.7812 [1.171875, 6.390625]|
|A/seed-1401/validation-teacher100.json|37/27/0|0.5781 [0.46875, 0.6875]|+1.2500 [-1.5628906249999999, 4.265625]|
|A/seed-1402/validation-gen2.json|33/28/3|0.5391 [0.4375, 0.640625]|+1.4375 [-0.90625, 3.8441406250000014]|
|A/seed-1402/validation-oldvalue.json|41/20/3|0.6641 [0.5390625, 0.78125]|+4.7500 [2.015625, 7.3125]|
|A/seed-1402/validation-previous.json|38/21/5|0.6328 [0.5234375, 0.7421875]|+2.5312 [0.359375, 4.921875]|
|A/seed-1402/validation-teacher100.json|31/30/3|0.5078 [0.3984375, 0.625]|+1.0781 [-1.5625, 3.75]|
|A/seed-1403/validation-gen2.json|32/27/5|0.5391 [0.4453125, 0.640625]|+3.0625 [0.84375, 5.375]|
|A/seed-1403/validation-oldvalue.json|40/23/1|0.6328 [0.5234375, 0.7421875]|+3.7031 [1.03125, 6.375]|
|A/seed-1403/validation-previous.json|27/34/3|0.4453 [0.3359375, 0.5546875]|+0.3125 [-2.546875, 3.25]|
|A/seed-1403/validation-teacher100.json|38/25/1|0.6016 [0.4921875, 0.7109375]|+1.2969 [-1.3125, 4.1875]|
|B/seed-1401/validation-A.json|33/30/1|0.5234 [0.4375, 0.609375]|+1.5938 [-0.43789062499999987, 3.71875]|
|B/seed-1401/validation-gen2.json|37/26/1|0.5859 [0.5, 0.671875]|+2.7188 [1.1089843750000001, 4.265625]|
|B/seed-1401/validation-oldvalue.json|36/26/2|0.5781 [0.46875, 0.6796875]|+3.3281 [1.0308593750000001, 5.5625]|
|B/seed-1401/validation-previous.json|36/27/1|0.5703 [0.453125, 0.6796875]|+2.5625 [-0.171875, 5.422265625000001]|
|B/seed-1401/validation-teacher100.json|33/26/5|0.5547 [0.4296875, 0.6796875]|+2.3281 [-1.15625, 5.875]|
|B/seed-1402/validation-A.json|36/28/0|0.5625 [0.4375, 0.671875]|+0.9062 [-2.6878906249999996, 4.46875]|
|B/seed-1402/validation-gen2.json|28/32/4|0.4688 [0.359375, 0.578125]|-1.3281 [-3.4378906249999996, 0.671875]|
|B/seed-1402/validation-oldvalue.json|40/21/3|0.6484 [0.5390625, 0.7578125]|+3.7969 [1.859375, 5.671875]|
|B/seed-1402/validation-previous.json|26/32/6|0.4531 [0.3515625, 0.5546875]|-0.8125 [-2.484375, 0.75]|
|B/seed-1402/validation-teacher100.json|30/32/2|0.4844 [0.3671875, 0.6015625]|+0.3438 [-2.203125, 2.78125]|
|B/seed-1403/validation-A.json|39/21/4|0.6406 [0.5390625, 0.75]|+0.5938 [-2.0941406249999996, 3.1410156250000014]|
|B/seed-1403/validation-gen2.json|35/27/2|0.5625 [0.4453125, 0.6796875]|+0.6562 [-2.046875, 3.171875]|
|B/seed-1403/validation-oldvalue.json|48/13/3|0.7734 [0.6875, 0.859375]|+7.6875 [5.125, 10.375390625000001]|
|B/seed-1403/validation-previous.json|34/27/3|0.5547 [0.4609375, 0.6484375]|+0.0938 [-2.625, 2.8125]|
|B/seed-1403/validation-teacher100.json|35/26/3|0.5703 [0.453125, 0.6875]|+3.2656 [1.25, 5.312890625000001]|
|B/seed-1401/holdout-A.json|94/90/8|0.5104 [0.4348958333333333, 0.5807291666666666]|+0.7969 [-0.5625, 2.1145833333333335]|
|B/seed-1401/holdout-gen2.json|101/79/12|0.5573 [0.4947916666666667, 0.6197916666666666]|+1.4844 [0.2916666666666667, 2.614713541666667]|
|B/seed-1401/holdout-teacher100.json|109/74/9|0.5911 [0.5260416666666666, 0.65625]|+2.1771 [0.6145833333333334, 3.6979166666666665]|
|B/seed-1402/holdout-A.json|69/114/9|0.3828 [0.3203125, 0.4505208333333333]|-1.2656 [-2.4114583333333335, -0.08333333333333333]|
|B/seed-1402/holdout-gen2.json|102/80/10|0.5573 [0.4895833333333333, 0.6276041666666666]|+1.5208 [0.17708333333333334, 2.8802083333333335]|
|B/seed-1402/holdout-teacher100.json|87/98/7|0.4714 [0.3984375, 0.5416666666666666]|-0.5104 [-1.8125, 0.7916666666666666]|
|B/seed-1403/holdout-A.json|101/80/11|0.5547 [0.4895833333333333, 0.6197916666666666]|+0.7708 [-0.557421875, 2.1095052083333337]|
|B/seed-1403/holdout-gen2.json|100/83/9|0.5443 [0.4791666666666667, 0.6067708333333334]|+1.8594 [0.5102864583333333, 3.2291666666666665]|
|B/seed-1403/holdout-teacher100.json|123/65/4|0.6510 [0.578125, 0.7161458333333334]|+3.6094 [1.8169270833333335, 5.416796875000001]|

## 検証・制約

{"validation": {"games": 1728, "unique_actual_states": 49013, "training_actual_overlap": 0, "adopted_start_overlap": 0}, "holdout": {"games": 1728, "unique_actual_states": 51664, "training_actual_overlap": 0, "adopted_start_overlap": 0}}

過去全履歴は既存全再生監査・checkpoint/history SHA参照で保全。新規履歴は全合法性・終局報酬・指定learner側target・pool割当を再生監査。A/B双方の全訓練訪問state和集合を評価開始状態から事前除外し実訪問も後監査。初期空盤面と評価decision局面は区別。固定policy SHA変更を拒否。CUDA別process6144→6150連続6 vs6144→6146保存＋追加4で、pool循環と6148過去policy交換をまたぎmodel/past/optimizer/RNG/pool/履歴bit一致。

相手別訓練pointは学習中policyへの結果で、固定相手への評価成績と混同しない。複数世代枠の各ゲーム数は少なく単一seedの偶然を一般化しない。今回の有限範囲で停止しmain推奨を変更しない。

## 再現CLI

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/check_ppo_pool_resume.py
for kind in A B; do
  for seed in 1401 1402 1403; do
    .venv/bin/python -m blokus.ppo_scale --seed "$seed" --resume "artifacts/ppo-6144/seed-$seed/iteration-6144.pt" --output "/tmp/pool-repeat/$kind/seed-$seed" --iterations 7168 --save-steps 6400 6656 6912 7168 --cap 850 --forbidden-prefixes artifacts/ppo-generation-pool/forbidden-prefixes.json --pool-config "artifacts/ppo-generation-pool/pool-$kind.json"
  done
done
.venv/bin/python -m blokus.ppo_pool_eval audit_eval
.venv/bin/python scripts/report_ppo_pool.py
```

既存runを上書きしない。GPU時間探索のbit再現は保証しない。研究checkpointは履歴をSHA参照し、過去履歴を複製しない。

## 結論

複数世代poolの改善は確認できなかった。freshのB対Aはpoint48.26% [43.83,52.69]、得点差+0.101 [-0.766,+0.929]。seed別pointは1401=51.04%、1402=38.28%、1403=55.47%。1402ではBの退行があり、validationの平均57.55%はfreshで再現しない。seed選別・checkpoint再選択・gate変更はしていない。

B対gen2はpoint55.30% [51.65,58.85]、得点差+1.622 [0.882,2.387]。B対100ms教師はpoint57.12% [52.69,61.37]、得点差+1.759 [0.809,2.641]。この固定基準への平均成績を、pool変更による改善と混同しない。教師戦はseed1402がpoint47.14%/得点差−0.510であり、seed差もある。Bの推論は約1.40ms、教師約69.33ms/手（100ms超過0%）で同時間比較ではない。B対Aでは約1.328/1.321msで、両方とも同じネット単体アーキテクチャ。

main gen2と既存推奨は維持する。今回の1/12ずつの循環配分には採用根拠がなく、無制限な延長は行わない。固定世代が弱すぎた、直近pastへの露出を減らした等は仮説であり、この実験だけで原因を特定しない。次段階でpoolを再検討するなら、年齢だけでなく凍結相手への実力差と汎化を事前に測る対照が必要。現条件を優先して保持し、細かなpool割合探索を自動で繰り返さない。

## 最終確認・記録の読み方

6runは全て追加1024iteration/16384局、計98304局で完走。生成＋更新＋保存wall3895.82秒（64.93分）、minibatch更新計226775。生成からfresh評価・監査の完了まで5296.26秒（88.27分、最終文書・push時間を除く）。CUDA別process再開smokeは別途実施し、6=2+4、pool循環/past交換込みbit一致。mainの学習条件・重みは変更していない。

validation1728局、holdout1728局のengine.is_legal（合法手列挙とは別経路）による手再生・終局得点/勝敗・prefix組/席・checkpoint SHA・時間集計・分離監査PASS。採用開始状態と評価実訪問状態の訓練和集合重複は0。追加監査で対照モデルの指定とgate算術も再確認し、保存済み成績は変更なし。全112tests PASS。最終CLI動作確認の1局は評価成績には含めない。

準備関数の流用でlock.jsonに残ったlatest(4096)/criterion_weightsは未使用の旧項目であり、今回の再開元はresume(6144)、選択規則は全条件共通最大iteration、進行gateはPLAN.md/decision.json。元の凍結lockは改変せずmetadata-notes.jsonに訂正を記録し、将来の準備関数では未使用項目を除去した。モデル・相手・評価条件の変更ではない。

過去履歴は参照のみ、今回新規分の履歴を保存。6400/6656/6912は予算打切り時の共通保存点として残し、成績による選択に使っていない。推論例:

```bash
.venv/bin/python -m blokus.terminal_play --model artifacts/ppo-generation-pool/B/seed-1401/iteration-7168.pt --human-seat 0
.venv/bin/python -m blokus.terminal_play --model artifacts/ppo-generation-pool/B/seed-1401/iteration-7168.pt --opponent artifacts/ppo-generation-pool/A/seed-1401/iteration-7168.pt
```
