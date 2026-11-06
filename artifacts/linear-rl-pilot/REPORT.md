# 非ニューラル強化学習の初回pilot

公開gen3/mainは変更しない。これはニューラルネットを用いない方策学習の探索的比較であり、強い公開モデルへの置き換え判定ではない。

## 方法と出発点

`blokus.linear_rl` は手作り特徴の線形和をsoftmax確率にして着手し、最終勝敗(+1/0/-1)からREINFORCEの解析勾配で重みを更新する。NumPyで確率と勾配とAdamを計算し、ニューラルネット・隠れ層・誤差逆伝播は使わない。既存エンジン・特徴生成モジュールを共有するため、配布環境としては既存のTorch依存が残る。GPU処理は不要。

10特徴は、駒のマス数、接続点の増減、相手接続点を覆う量、中心寄り度、開始点からの広がり、相手接続点への辺接触量、自分の接続点数、候補駒の角の空き、1マス駒か、相手の駒への辺接触量。着手間で値が同じ2特徴は除いた。残り駒数15〜21/8〜14/0〜7で重みを分け、合計30係数。

初期値は各段階とも既存heuristicと同じ。ランダムな評価や一様ランダム着手からの開始ではない。学習中はsoftmaxで探索し、評価時は最大評価の着手を選ぶ。

3seed × 256iteration × 16games = 12,288局。学習率0.03、Adam、勾配ノルム上限5。相手はbatch開始時の自己50%、64iterationごとに保存する過去自己25%、固定初期heuristic25%。各相手で先後を均等にした。終局報酬だけを使い、教師の手は使わない。最後のcheckpointを固定し、中間成績による選択・設定変更はしない。

既存`blokus.cem`は対局結果を用いた重みの直接探索で、今回の着手確率に対する方策勾配とは別方式。

## 結果

学習生成・更新・保存の合計21.57秒（初回importやnativeビルドは含まない）。各seed4,096局の短期pilot。64個の新規ランダム4手開始局面、先後交換、3つの固定学習モデルを合算。対戦中の各着手は独立の座標判定で合法性検証。371,740個の訓練訪問状態と評価の開始・途中状態の一致は0件。

| 相手 | 対局数 | 勝ち1・引分0.5の平均 | 95% CI | ゲームの平均得点差 |
|---|---:|---:|---:|---:|
| 学習前の手作り評価 |384|56.38%|51.30〜61.33%|+0.60（CI −0.15〜+1.34）|
| 公開gen3 seed1751401 |384|13.15%|9.90〜16.67%|−8.41（CI −9.23〜−7.59）|

CIは開始局面ごとのクラスタbootstrap。3学習seedを固定した条件付きの幅であり、任意の新しい学習seedや全開始局面への一般化を保証しない。相手gen3は既定の1モデル。多重比較補正・別holdoutは行っていない探索的pilot。gen3を上回った結果ではない。

16保存開始局面×5反復を交互測定したCPU着手計算の中央値：線形0.0828ms、gen3 0.3949ms。双方native特徴生成、1thread。約4.77倍の差だが、学習効率や終盤を含む全分布での速度比ではない。

## 検証・再現

softmax方策勾配の有限差分、初期評価とheuristicの一致、保存再開、native/Python学習結果一致、passをテスト。既存テストを含む全90テストPASS。CLIのAI対戦完走と人間入力EOF終了も確認した。生結果はPLAN.json、train.json、evaluation.json、summary.json、3seedのstate/policy JSON。

```bash
# 別worktreeまたはcloneで実行。optional nativeを使う場合
python -m blokus.ppo_native build
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. python scripts/linear_pilot.py --output artifacts/my-new-pilot
# 学習CLI（Python特徴生成でも動く）
python -m blokus.linear_rl --output artifacts/my-linear --iterations 256 --seed 3101 --backend native
# 既存terminal UIで人間対戦（先手なら0、後手なら1）
python -m blokus.linear_play --checkpoint artifacts/linear-rl-pilot/policy-3101.json --human-seat 0 --backend native
```

今回の結果を踏まえた次段階は、(1)到達可能な空き領域/残り駒の配置可能性などの特徴を追加、(2)局面価値を線形TD学習し先読みの葉に使用、(3)同じ思考時間で探索なし/あり/gen3を比較する、の順を提案する。特徴追加・TD価値学習・探索結合は今回は未実施。非線形な表現が必要なら、局所パターンの値を表に保存して学習するn-tuple方式も候補となる。
