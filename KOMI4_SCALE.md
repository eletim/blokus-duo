# コミ4点差AI・5世代拡大学習（学習完了・最終比較中）

新規学習用11,000一意対局を目標とする追加キャンペーン。前回pilotの72対局や旧データは、この局数に含めない。
閲覧: http://100.95.30.96:8766/ の05節「拡大学習」。生成・学習使用済み一意対局・局面・除外・epoch・更新・実学習時間・総経過時間を別表示する。

## 計画と測定

- 5世代×2,200新規対局、3モデルseed。各世代200予備条件を事前固定。使用可能対局が2,000未満なら停止・原因確認し、水増ししない。
- 学習生成は同世代自己対戦40%、gen4/前回非線形/旧ridge各20%。同じ決定的自己対戦を先後交換して増やさない。
- 8/10/12手開始。25%は一様合法手、75%は凍結方策上位16手から一様抽選。純粋な正規開始とは区別する。全乱数条件を事前保存。
- 生成150ms・幅4・最大深さ4。実測48局は8並列で7.15秒、第1世代の学習データとして保存し再生成しない。
- 第1世代は2,200一意対局/56,099局面、生成約251秒、3seedで31epoch/3,410更新/実学習4.49秒。小型94,417パラメータなのでGPU計算自体は速い。時間消費を目標としない。
- 第2世代以降も新方策データを生成し、直近世代と過去の新規世代をloss重量50:50で使う。全局面を各epochで巡回し、対局長で重みを正規化。
- GPU float32/AdamW lr0.0003、validationでLR低減、最大50epoch・最低8epoch・改善なし6epochで停止。更新前checkpointも候補に残し、改善しない更新を採用しない。
- 新validation256局を固定。未知test256局は最終選択後のみ評価。過去testを未知として再使用しない。
- 各世代の比較は別条件64局。最終確認は新128prefix×先後×旧ridge/前回非線形/gen4＝768局、正規開始3seed×先後×3相手＝18局。全員500ms・幅4・最大深さ6。
- gen4の途中評価はコミ0で学習された凍結WDL。研究用終局判定のみコミ4に対応。本番モデルは変更しない。

後手+4、先手視点の終局差はscore0−score1−4、後手は反転。全置きボーナスを含む点差と残りマス差を区別する。コミ定数の平行移動だけで手の順位は変わらない。

## 監督と再開

`blokus-komi4-scale.service`（systemd user）で有限実行。CPU4–11の8コア、CPUQuota800%、MemoryMax24GiB、Nice5、初回24時間以内。元8765、gen4 frozen、人間棋譜は保全。

- 稼働確認: `systemctl --user status blokus-komi4-scale.service`
- ログ: `journalctl --user -u blokus-komi4-scale.service`
- 実台帳: `artifacts/komi4-scale/operations.jsonl`、最新確認: `status.json`
- 全棋譜: `games/*.json.gz`、固定条件: `plan.json` / `tasks.json`
- 学習使用根拠: `features/*.json`、配列cache: `features/*.npz`
- モデル/選択・再開checkpoint: `models/`、学習曲線: `training/`

実行入口は `PYTHONPATH=. /home/eletim/blokus-duo-ai/.venv/bin/python scripts/komi4_scale.py run`。
再開前にservice/PIDと台帳を確認する。同時起動は `.run.lock` で拒否。既存棋譜はtask/modelハッシュ照合でスキップ。学習はoptimizer/scheduler/CPU・CUDA・NumPy RNGを含むepoch checkpointから再開する。失敗を成功とみなさず、原因確認後に必要部分だけ続行する。

## 学習完了時点

5世代で新規11,000一意学習対局・281,197局面、3seed合計141epoch・44,523更新・実学習56.21秒。validationのみで3seedとも第5世代を選択。新しい保留test256対局・6,475局面でMAEは4.045/3.988/4.119点（前回非線形4.563/4.561/4.522、ridge5.627）。対局群bootstrapによる前回モデルとの差の95%区間は各seedで0を含まない。ただし教師は固定混合方策の実終局結果で、最適プレイの真値ではない。対戦性能は最終比較待ち。

生成済み数と学習使用済み数は異なる。validation/testは学習局数に含めない。比較結果と精度評価は最終完了後に追記する。現時点で強化達成とは扱わない。
