# 固定新世代PPO：独立標本再確認・世代別提供

追加学習なし。fd527caのvalidation選択済み1401=3072、1402/1403=4096を固定。seed1620000の新96prefix・先後交換・3対照・3seed、計1728局。同一prefixを全相手で共有。訓練全訪問状態と過去評価prefix/開始状態を評価前に除外し、評価後も実訪問を監査した。固定モデル条件での標本再確認であり、秘書による独立実装監査とは呼ばない。

## 再確認結果

|相手|point [95% CI]|平均得点差 [95% CI]|PPO/相手 実平均ms|相手予算超過率|
|---|---|---|---|---|
|previous|0.6372 [0.6067708333333334, 0.6666666666666669]|+3.9583 [3.2030815972222233, 4.6649739583333325]|1.442/1.271|適用なし|
|oldvalue|0.6337 [0.5963541666666664, 0.6692708333333331]|+2.9184 [2.095486111111111, 3.7048611111111107]|1.286/4.583|0.067%|
|teacher100|0.5217 [0.4791666666666667, 0.5633680555555555]|+0.7674 [-0.23658854166666635, 1.7483940972222234]|1.414/69.588|0.000%|

pointは勝1/引分0.5/負0。2座席と3seedをprefix内平均して96prefix単位bootstrap（4000回）。固定モデル条件のCIであり更新seed母集団CIではない。ネット単体にはdeadlineがなく、同時間比較ではない。教師cost/refine予測は各試合cold。時間計測を学習・監査・敗局解析から分離し、計測後に再生監査した。超過は全act実時間が相手の6/100ms上限を超えた割合。モデル初期化/warmupは従来通り別。

|seed/相手|勝/敗/分|point|得点差 [95% CI]|
|---|---|---|---|
|1401-previous|103/81/8|0.5573|+1.5677 [0.46875, 2.7033854166666678]|
|1402-previous|112/71/9|0.6068|+3.1198 [1.8854166666666667, 4.302213541666667]|
|1403-previous|140/45/7|0.7474|+7.1875 [5.708333333333333, 8.666796875]|
|1401-oldvalue|115/70/7|0.6172|+2.5677 [1.4375, 3.6773437500000012]|
|1402-oldvalue|106/72/14|0.5885|+2.3333 [0.9010416666666666, 3.8023437500000012]|
|1403-oldvalue|132/57/3|0.6953|+3.8542 [2.5675781250000003, 5.104296875000001]|
|1401-teacher100|94/91/7|0.5078|+0.5729 [-1.026171875, 2.0835937500000012]|
|1402-teacher100|94/94/4|0.5000|+0.2969 [-1.0313802083333332, 1.6251302083333339]|
|1403-teacher100|104/82/6|0.5573|+1.4323 [-0.22929687499999996, 3.083463541666667]|

## main提供条件

事前gate clear_regression=False、PPO CLI既定世代=gen2。3seed/checkpointは再選択しない。旧2048と既存value/2ply・100ms教師を残す。グローバル旧AIデフォルト不変、PPO seed既定1401も維持。新世代推論checkpointは履歴/RNG/optimizer不要、研究履歴の複製なし。CPU/CUDA export前後の一致・新clone検証はexport-check.json/fresh-clone.jsonを参照。

## 少数敗局の記述的診断

各seed×対2048/教師100msの最初の敗局1局、計6局。絶対ply<14/14〜27/28以上を序盤/中盤/終盤とし、各期最初の自手番（pass以外）を採取。選択バイアスの強い標本であり頻度推定・因果推定には使わない。参照は固定value alpha=.5/gate24、heuristic上位128root＋実際の手、相手上位3応手の完了2plyで、真のゲーム価値ではない。

|phase|標本数|平均参照値gap|平均heuristic順位|正gap数|
|---|---|---|---|---|
|opening|6|0.300|16.33|3|
|middle|6|0.933|17.50|4|
|ending|3|0.298|1.67|3|

- seed1401 vs previous: seat0, 得点差-17, 最終残駒サイズ 自/相手=[[3, 4, 4, 4, 4, 5, 5], [3, 4, 5]]。prefix=1777e3b8b0021cc09051d1af5f1893e731ad53ac94e909717db386036c821b27。

- seed1401 vs teacher100: seat0, 得点差-14, 最終残駒サイズ 自/相手=[[3, 4, 4, 5, 5, 5, 5], [3, 4, 5, 5]]。prefix=1777e3b8b0021cc09051d1af5f1893e731ad53ac94e909717db386036c821b27。

- seed1402 vs previous: seat0, 得点差-6, 最終残駒サイズ 自/相手=[[3, 3, 4, 5], [4, 5]]。prefix=1777e3b8b0021cc09051d1af5f1893e731ad53ac94e909717db386036c821b27。

- seed1402 vs teacher100: seat0, 得点差-15, 最終残駒サイズ 自/相手=[[3, 4, 4, 4, 4, 5], [4, 5]]。prefix=1777e3b8b0021cc09051d1af5f1893e731ad53ac94e909717db386036c821b27。

- seed1403 vs previous: seat0, 得点差-3, 最終残駒サイズ 自/相手=[[3, 4, 4, 4, 4, 5], [4, 4, 4, 4, 5]]。prefix=31d1377a009af65edbfc357f90b6b42961be320c95150affc42250392feff58a。

- seed1403 vs teacher100: seat0, 得点差-3, 最終残駒サイズ 自/相手=[[3, 3, 4, 4, 5, 5, 5, 5], [4, 4, 4, 4, 5, 5, 5]]。prefix=1777e3b8b0021cc09051d1af5f1893e731ad53ac94e909717db386036c821b27。

6敗局は2prefixを共有し、独立6標本ではない。序盤では参照2plyと同値の手も3/6あり、序盤順位だけを主因とは断定できない。中盤は平均参照gap0.93で6局すべて自分の合法手数が相手より少なく、特に1401対2048は4対310だった。終盤サンプルは3局のみ（他は28ply以前に自分の合法手が尽きる）で、平均gap0.30と小さく、全6敗局で3〜5マス駒が残った。領域・駒順の問題を疑う観察だが、prefixの先後不利や相手の選択も混在する。敗局終盤の合法手減少・残駒、序中盤の参照値との不一致をloss-analysis.jsonに保存。別の手を選んだ場合の完走比較や勝局との対応対照はないため、領域確保・駒順・読み不足のいずれが主因かは確定できない。holdoutを訓練へ転用せず、重み・候補・設定を変更していない。

## 次実験の推奨

第一候補は(1)同条件の有限継続。前段階で学習量増加による独立holdout改善があり、今回は凍結再確認のため強化要因を追加していない。3072/4096のseed差があるので既存選択モデルを比較基準として保持し、事前に上限とvalidationを固定する。(2)複数世代poolは旧世代への耐性の仮説として有用だが、今回の少数敗局だけでは必要性を実証していない。(3)小予算探索は教師との実時間差が大きく、ネット単体の利点と推論費用を損なう可能性があるため、別の固定モデル対照が必要。無期限継続は推奨しない。

## 再現CLI

```bash
.venv/bin/python -m blokus.ppo_generation_eval audit
.venv/bin/python scripts/analyze_ppo_generation_losses.py
.venv/bin/python scripts/check_ppo_gen2_export.py
.venv/bin/python -m pytest -q
mkdir -p /tmp/ppo-gen2-repeat
cp artifacts/ppo-generation2/{lock,prefixes}.json /tmp/ppo-gen2-repeat/
.venv/bin/python -m blokus.ppo_generation_eval run --root /tmp/ppo-gen2-repeat
.venv/bin/python -m blokus.ppo_generation_eval audit --root /tmp/ppo-gen2-repeat
# 公開推論
.venv/bin/python -m blokus.ppo_play --generation gen2 --model-seed 1401 --human-seat 0 --device cpu
.venv/bin/python -m blokus.ppo_play --generation gen2 --opponent ppo --opponent-generation 2048 --device cuda
```

prepare/run/auditは順次実行する。過去prefixを再生成するのではなく保存prefixes.jsonとlock.jsonから固定モデル・同一fixtureで再生する。教師の時間制御探索はOS/実費用に依存し、対局結果のbit再現は保証しない。

## 統合・最終検証

main a96d280347930ef3d8f916725f00b6474d4884d7（通常cherry-pick 22801aa、元80a5179）をpush。main59テスト、CPU専用推論6テスト、CPU/CUDA各13局CLI再生PASS。研究全108テストPASS。3seed×CPU/CUDA×26局面のexport前後入力/logits/value/greedy出力bit一致。旧2048との世代対戦も完走。

追加学習0秒/0更新。1728対局の計測時間合計714.5秒。監査・文書整理の途中で記録した全体経過18.4分。compute.jsonの経過は最終pushを含まない。今回の追加保存は評価・設定・監査と軽量推論exportのみで、過去の圧縮訓練履歴を複製していない。
