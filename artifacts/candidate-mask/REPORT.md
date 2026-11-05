# 候補セルmaskの汎化対照（2026-11-05）

結論: 候補の正確な配置を入力・勾配で区別できたが、対戦進行gateは両表現とも未達。mainと既存value/2ply推奨を維持。対戦validation/holdoutは作成していない。gate変更なし。

## 固定条件と独立性

基点312e9dc、事前条件はPLAN.md。前回poolの3600独立元ゲームだけを再利用。3foldの訓練実数2380/2388/2384ゲーム、4489/4508/4503 roots。新bank seed1330000–1330359の360元ゲームから356ゲーム672 rootsが有効、各fold119/117/120ゲーム。1ゲーム最大2roots、相手3種×座席×進行度を共通にした。旧bank/holdoutは学習・選択に含めない。

固定教師のdepth4-depth2補正、同深度で完了したtop4+random2候補、Huber、alpha .5、固定value/応手順位を保持。教師は幅制限探索の値で真のゲーム価値ではない。候補集合top128からの選択偏りは残る。既知候補だけをcompact化して処理し、mask falseのpaddingは損失から除外。raw MSEはalphaを掛けない予測、選択損失はalpha .5適用後のteacher regret。

両表現とも1000更新×3fold×3更新seed1101/1102/1103、AdamW lr.001、weight_decay0、batch32 root。初期seed1100、共通パラメータ全て同一。maskの追加channelだけ独自初期値。データ・ミニバッチ乱数も共有。summary58457、cells58529パラメータ（+72、0.123%）。cellsは候補ごと盤面CNNを計算するため演算量は同一ではない。対称変換は使用しない。今回の固定1000更新後に結果を見て更新数を増やしていない。

## 未使用ゲームの結果

ゼロ補正raw MSE 6.91439、選択損失0.540837。

|表現|raw MSE|ゼロ比改善CI95%|選択損失|ゼロ比改善CI95%|
|---|---:|---|---:|---|
|summary|6.45957|[0.016114666090871865, 0.8792920329095489]|0.52051|[-0.024880017379947588, 0.06514638077174682]|
|cells|6.45132|[0.022942032989407505, 0.8899189990470558]|0.53798|[-0.04379240148927501, 0.04835104867606684]|

maskの選択損失改善は約0.53%、summaryは約3.76%。maskはfold2でMSE/選択損失ともゼロより悪化、seed1102も選択損失が悪化。summaryも全fold5%改善・選択損失CIのgate未達。maskがsummaryより良いという根拠もない（paired summary−cells 選択損失CI [-0.042265539125919564, 0.0055518822353821645]）。学習データMSEはsummary4.548/cells4.534で、新ゲームでは約6.45。訓練誤差低下は汎化改善の証拠にならない。

CIは同一ゲームのrootを平均し、3更新seedもゲーム内平均後に4000回game bootstrap。foldの訓練集合は重複するので9runを独立標本とは数えない。fold別・seed別の値はcomparison.json、全ゲームledgerは各result.jsonに保存。CIはこの訓練集合・学習済みモデルに条件付けたもの。

## 検証と費用

全89テストPASS。実在pool内の同一39要約・異なる配置例をcollision.jsonに保存し回帰テスト化。mask入力と追加conv channelに非ゼロ勾配、配置による出力差を確認。共通初期値と+72パラメータ、未知mask、終局保護、補正が最終選択に残ることも検証。ゼロ初期出力は両者同じで、区別テストは非ゼロ出力重みを使う。

CUDA別processで各表現「40連続」対「20保存+追加20=計40」: model/optimizer/Python/NumPy/Torch/CUDA RNGすべてbit一致。元3960対局の合法再生・終局得点、7422ラベルroot、75root教師値再計算、split hash、集計/CI/gate監査PASS。これは実装側監査であり秘書による独立実装監査ではない。

保存した両代表checkpointについてNumPyによる別集計も119ゲーム226rootsずつ一致（prediction-audit.json）。

訓練/評価の実際のラベルroot hash重複0、元ゲームの全手順hashも3960件unique。ただし元履歴全体ではply8以後155状態が共通（147はply8–15、8は16–23）。評価root3件は訓練元履歴中の**ラベル化していない**状態と一致し、逆方向は13件。これら未採用履歴を学習には使っていない。全履歴が完全非重複とは主張しない（trajectory-overlap.json）。

新bank生成・データ化132.14秒、GPU主学習18,000更新19.82秒（評価・保存込み）、上限1200秒未満。予算が余ったことを理由に更新/データを増やさなかった。RTX5090、PyTorch2.7.1+cu128。推論計測は生成/学習/監査と分離。128候補のforward+同期はsummary0.120ms/cells0.135ms、追加peak割当1.09/2.45MiB。合法手生成・順位・pack・転送・同期込み1.314/1.340ms（単一診断root、2ply本体は含まない）。共通packingのためsummaryにもmask構築費用が入る。全試合25ms性能や棋力を示す数字ではない。

## 判断材料

配置情報の欠落は修正できたが、この規模・容量・更新数では選択損失の汎化は改善しなかった。入力衝突の誤差下限自体が小さく、これだけで成功するとは言えないことを確認した。幅制限教師への依存、top4+random2の観測偏り、教師残差と実戦最終得点の不一致、相手の適応、ゲーム間の分布差は残る。全候補への汎化や実戦強さも未確認。

過去のデータ量・損失・正則化と今回の入力対照を合わせ、root補正の小変更をこのまま続ける根拠は乏しい。この系統はいったん区切る判断を推奨する。次系統は実対局の最終結果を使う自己対戦policy/value学習を候補とし、相手poolを固定baselineも含めて多様化、ゲーム単位分割と学習前対照を維持する。今回の結果は自己対戦の成功を保証しない。秘書が移行を判断するまで追加開発しない。

## 再現

```bash
cd /home/eletim/blokus-duo-ai
# 保存した新データで全18条件を再現（結果既存ならskip）
.venv/bin/python -m blokus.mask_experiment
# 単独学習、1000は総更新数。checkpoint保持先は任意。
.venv/bin/python -m blokus.mask_train --size 2400 --fold 0 --seed 1101 --condition cells --steps 1000 --output /tmp/mask-replay
.venv/bin/python scripts/check_mask_resume.py
.venv/bin/python scripts/audit_mask.py
.venv/bin/python scripts/audit_mask_overlap.py
.venv/bin/python scripts/audit_mask_predictions.py
.venv/bin/python -m pytest -q
# 他runを停止して単独計測
.venv/bin/python scripts/profile_mask.py
```

元対局は時間制御なのでseedのみで完全再生成される保証はない。保存source-games/teacher-labelsを再生するのが正確な再現。mask_prepareで保存data-2400からcompactデータを再作成可能。新生成はmask_data（既存データがあればresume/skip）を使う。代表checkpointはfold0/seed1101各表現1個のみ保持し、推奨モデルではない。CandidateAgentは既存CorrectionAgentの最終値補正経路を共有し、合法手生成・mask作成・GPU同期・推論もdeadline内に含める。gate未達のためCLI対戦評価は実施していない。

実行状態: 有限runすべて終了、計画記録から約8分（execution.json）。残存学習/生成プロセスなし。
