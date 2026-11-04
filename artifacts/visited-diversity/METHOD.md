# 実戦局面の多様性とcheckpoint選択

基点9e76141。エンジンと既存valueは固定。絶対値smoothL1(beta=.3)のみ使用し、損失種類の追加探索はしない。実験コードは blokus/visited_{data,train,eval,experiment}.py。

## 先に固定した比較

pilotはwide512root×4完走、dense128root×16完走。同じ実戦収集集合の残駒数層化subsetをdenseへ使う。両条件の候補数・相手応手数は同じ、完走数の積を等しくする。実際の継続ply数・CPU時間は別記し、厳密な等時間とは呼ばない。全量へ進む条件は、共通devでlive意思決定regretの3seed平均が既存valueより10%以上低下し、live選択モデルの新validation対value平均が+0.25点以上。pilotを通過した場合のみ、実測から35分のラベル生成予算内にroot数を調整（上限wide10000/dense2500）。

実ゲームは既存value、6ms時間制御。初期ランダム2〜6ply、その後8%の確率で合法ランダム着手を混ぜる。両者合計残駒24以下、時間内に4候補以上を評価し終えたrootだけを採集し、最初の4候補と各上位3応手のleafを保存する。期限後に返ったleafは除外。同一意味状態hashで重複除去、1ゲーム最大12root。これは実戦全分布ではなく、4候補を計算できた中終盤の条件付き分布である。探索幅が4を超える部分や、少ない合法候補しかない最終盤は学習対象外。

継続poolはstatic/static・value/value・value/static・static/valueの4組。既存value固定、4×3探索で終局まで進め、5%ランダム着手を含む。候補と応手の順序は保存された実探索の順序。root hashから継続seedを決め、同rootの候補・応手間で同じ完走番号のpool/各ply乱数を共有する。状態が違うため実際の着手は異なる。同じrootのdense先頭4完走はwideの4完走とbit一致。16完走まで増やした安定度と独立8対8の一致率を診断する。

3更新seed601/602/603は同じデータ・候補順序・継続ラベル・初期既存value重み・validation/holdoutを共有する。更新のminibatchと対称変換の乱数のみ異なる。したがって独立した3データ収集試行ではない。

## checkpoint選択と分離

各更新列から2checkpointを選ぶ。rawはネットワークの値そのもの、liveは実戦と同じalpha=.5の静的値混合および正確な終局値上書き。各候補の相手応手最小値を取り、同率候補を一様選択した期待意思決定regretを最小化する。同regretは差分MSEで比較。ラベルdevは独立seed430000の256root×8完走。同一ゲーム内から複数rootを採るため、256独立対局ではない。

pilot対局validation410000/16prefix、全量validation411000/32prefix。方式×selectorを3seed・2基準相手の平均得点差で固定。重みが完全一致したcheckpoint同士は同じ対局結果を再利用し、時間揺れをselector差と誤認しない。新holdout450000/96prefixは選択後のみ実行、選択方式の3seed対既存value・非学習探索と既存value対非学習探索だけ。各prefix先後交換。CIはprefixを単位とし、3seed総合はseed/prefixの交差bootstrap。96×3を独立prefixとは見なさない。旧holdoutは選択に使わない。

思考予算は共通6ms、実消費時間をpaddingで一致させたものではない。期限超過回数、平均/p95、探索node数を保存する。時間依存の探索候補は同じseedでも実行環境により変わるため、完全再生には保存済みデータ・prefix・ラベルを使う。

別の開発seed400000/8prefixでは、既存valueの重みを一切変えず6/25/100msを比較。学習効果の比較とは分離し、この小規模曲線でcheckpointを選択しない。

## 再現

既存成果を上書きしないよう、新しいcheckoutで artifacts/visited-diversity がない状態から全手順を実行する。CPU6worker・CUDA1枚を使い、計測と他GPUジョブを重ねない。

```bash
.venv/bin/python scripts/prepare_visited.py
.venv/bin/python -m blokus.visited_experiment run
.venv/bin/python scripts/verify_visited.py
.venv/bin/python scripts/report_visited.py
```

保存済みデータだけから個別再学習する例（新規出力先）:

```bash
.venv/bin/python -m blokus.visited_experiment train --data artifacts/visited-diversity/full/wide.npz --dev artifacts/visited-diversity/dev.npz --seed 601 --steps 6000 --seconds 120 --output artifacts/visited-retrain
# 保存再開は追加更新数。元と同じデータhashを要求する
.venv/bin/python -m blokus.visited_experiment train --data artifacts/visited-diversity/full/wide.npz --dev artifacts/visited-diversity/dev.npz --seed 601 --steps 400 --seconds 30 --resume artifacts/visited-retrain/latest.pt --output artifacts/visited-retrain
```

全量GPU主学習はpilot6モデル各90秒、full6モデル各120秒の最大計21分。実測はsummary.jsonへ保存。処理は有限更新・有限対局で終了する。中断時は各完了match結果、各モデルlatest.pt、ラベル生成完了ファイルを保持する。run全体の自動再開は未実装で、部分成果から個別train/evaluationを再開する。期限超過回復は既存BudgetAgentの費用見積りclipで維持。

固定済み最終7対戦だけの再実行（時間依存のため着手/成績のbit一致は要求しない。checkpoint再選択は禁止）:

```bash
.venv/bin/python scripts/replay_visited.py --output artifacts/visited-replay.json
```
