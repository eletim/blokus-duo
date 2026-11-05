# 応手幅の原因切り分け

基点f65bdbf、固定重み `artifacts/value-search/compact/seed-201/predictive-best.pt`、alpha=.5/gate24。追加学習なし。旧engine/value_search/deep_searchを変更せず、新node診断・幅だけを変えるReplyAgent・cold評価器を追加した。診断後・実装前の根拠はPROGRESS.md。

## 局面と参照

独立16ゲーム、seed620000+game、毎手heuristic上位8から一様ランダム。各ゲームから中盤1・終盤1。中盤は18/24ply目を目安にゲーム長の半分までで制限。終盤は末尾から探索して合法root数2〜8の最後の状態。終局直前への偏りがあり、難しい終盤全体の代表ではない。ゲーム着手列と状態hashを保存。標本単位はゲームであり32rootを32独立ゲームと扱わない。mid/end別に各16ゲームbootstrap4000回。

中盤参照はroot heuristic上位24、相手上位12、その後幅6、4plyの選択minimax。上限50000node、未完了参照ならrunはエラーとなりラベルに使わない。これは最終ゲーム結果の厳密解ではなく、同じvalueの深い近似。root24外・応手12外・以降幅6外の正しい最善手は不明。候補値も全てこの参照に条件付く。

終盤参照は全合法手・終局までの全探索、同50000node上限。pass・bonus込みの真の得点差。保存された16終盤全てを独立座標checker/別遷移/別得点の全探索で値と全最善手集合まで照合した。

## 指標

rootごとに浅い2ply葉値と参照継続値を同じ相手応手集合に保存する。

- 応手脱落: 参照値最小の応手がheuristic上位3に存在しない率。各root平均をまずゲーム内で取り、それを16ゲームで平均。差は`min(reference[:3])-min(reference)`。
- root脱落: root上位6の中に参照最善手が存在しないゲーム率と最善値損失。
- 選択損失(regret): 参照最善値−選択手の参照値。旧方式診断は**共通root24上の上位3応手/2ply規則**。実25ms旧探索の全候補や全合法rootに対する真のregretではない。
- 葉MAE: 2ply後状態のvalueと、同状態から参照継続した値の差。中盤では近似参照との差、終盤では真の最終値との差。このMAE・候補脱落損失・regretを足して因果分解したとは主張しない。

固定2048node上限でroot6/24 × 応手3/12 × 深さ2/4の8条件。nodeは訪問状態と評価leaf、TTなし、時間打切りなし。depth1はleafを一括評価し、その分を事前にnode予算へ計上。完了rootのみ採用、未完了root破棄、完了0ならheuristic先頭。上限は共通だが、浅い条件で余ったnodeを消費するpaddingはしない。制限以上にnodeを使わないテスト、同じ入力・上限で全結果一致を検証。

## 実戦変更と評価

変更は相手応手幅だけ。新validation621000/32prefixで幅6/12を既存上位3と共通25msで比較し、得点差最大の**幅6**を選択・selection.json保存。その後にfresh622000/96prefixを生成し先後交換192局を実行。既存holdoutも今回holdoutも再選択に使わない。

両者ともTTは使用しない。各試合前に適応cost_estimate/batch_reserve/expand_estimateを初期値へ戻す。モデルは保持し、試合統計カウンタだけ累積。phaseごとに新instance。GPUモデル初期化・warmupは両者とも思考時間外、探索中のrank/合法手生成/転送/推論は25ms内。以前のTT warm-cache成果は一切変更しない。時間計測中に診断・監査・テストは走らせない。per-moveのms/node/完了root/fallbackを保存する。

区間はprefix対単位bootstrap。勝敗は個々の192局を表示するが、区間算出時は先後2局を独立標本にしない。共通25msは実消費時間の完全一致やhard deadline保証ではない。node診断の着手は再現可能、wall-budget対局の完了範囲は実行速度で変化する。

## 再現コマンド

新規checkoutの既存成果は保全して別ディレクトリへ退避してからrunする（既存結果を自動上書きしない）。診断スクリプトは固定出力パス。保存済みselectionの再対戦は別outputに書ける。

```bash
.venv/bin/python scripts/diagnose_replies.py
.venv/bin/python scripts/summarize_replies.py
.venv/bin/python -m blokus.reply_experiment run
.venv/bin/python -m blokus.reply_experiment tournament --prefix-file artifacts/reply-diagnostics/holdout-prefixes.json --output artifacts/reply-replay.json
.venv/bin/python -m blokus.reply_experiment play
# 全時間計測終了後
.venv/bin/python -m pytest -q
.venv/bin/python scripts/audit_replies.py
```

設定・source hashはconfig/run-summary/verification、重みhashは全評価に保存。診断の32条件標本を2回ずつ再実行し固定node結果完全一致、全評価着手/終局/符号/CI/時間/完了候補数/選択最大を監査。開発ゲーム中終盤とvalidation/holdoutのhash一致なしも確認する。類似局面や分布の独立性はhashだけで保証しない。
