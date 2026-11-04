# Blokus Duo GPU AI 初版

Blokus Duoのルールエンジン、CLI対戦、GPU学習・評価の初版です。開発・学習環境はE-frontierの `/home/eletim/blokus-duo-ai`。

```bash
git clone https://github.com/eletim/blokus-duo.git
cd blokus-duo
```

以下のコマンドはclone先で実行してください。学習済みモデルと評価結果も同梱しています。外部の公式ルールPDFは同梱せず、本文の公式リンクを参照します。

## 環境とセットアップ

E-frontier、NVIDIA RTX5090 32GB、driver 595.91.07。専用 `.venv` はuv管理のPython 3.12.14を使用。

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python 'torch==2.7.1' --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python numpy pytest
.venv/bin/python -m blokus gpu-check
.venv/bin/python -m pytest -q
```

実際の導入バージョンは `requirements.lock.txt`、GPU検証は `artifacts/gpu.json`。
PyTorch公式の[Blackwell/CUDA12.8対応告知](https://pytorch.org/blog/pytorch-2-7/)と[導入手順](https://pytorch.org/get-started/locally/)を確認しています。

## 対局

```bash
# 人間が先手X、学習AIが後手O
.venv/bin/python -m blokus play --checkpoint artifacts/run/latest.pt
# AI同士（CPUでも可）
.venv/bin/python -m blokus play --players heuristic random --device cpu --seed 7
# 学習AI対heuristic
.venv/bin/python -m blokus play --players neural heuristic --checkpoint artifacts/run/latest.pt
```

盤面は行・列とも0始まり。人間は表示される合法手一覧のindexを入力します。各手にはpiece/orientation/row/colと配置セルを表示。合法手なしなら自動pass。開始点はX=(4,4)、O=(9,9)。CLIは大量の合法手をテキスト表示するためGUIほど快適ではありません。終了はCtrl-C。

## 学習と評価

```bash
# 60ゲーム教師あり初期化→自己対戦。300秒または200ゲームで停止
.venv/bin/python -m blokus train --games 200 --seconds 300 --imitation-games 60 --seed 42 --output artifacts/run
# optimizerとPython/NumPy/PyTorch/CUDA RNGを復元して続行
.venv/bin/python -m blokus train --resume artifacts/run/latest.pt --games 50 --seconds 120 --imitation-games 60 --output artifacts/run
# 最初から自己対戦のみ
.venv/bin/python -m blokus train --imitation-games 0 --games 50 --seconds 120 --output artifacts/selfplay
# 各相手に20局（先手10/後手10）、勝敗引分と平均得点差・各対局の最終スコア
.venv/bin/python -m blokus evaluate --checkpoint artifacts/run/latest.pt --games 20 --seed 1000 --output artifacts/trained_eval.json
.venv/bin/python -m blokus evaluate --checkpoint artifacts/run/untrained.pt --games 20 --seed 1000 --output artifacts/untrained_eval.json
.venv/bin/python -m blokus evaluate --agent heuristic --games 20 --seed 1000 --device cpu --output artifacts/heuristic_eval.json
```

評価の既定出力は `artifacts/evaluation.json` で、親ディレクトリは自動作成します。評価gamesは正の偶数。連続する2局で同じseedを使い先後を交替。探索や温度samplingなしのpolicy argmax（同点のみseed付き抽選）。randomは全合法手から一様選択。学習の停止時刻はゲーム境界で判定するため最大1ゲーム分超過し得ます。模倣終了モデルは `imitation.pt` として別保存します。学習のデフォルトは200局/300秒（模倣60局）。初期モデルは `untrained.pt`、最新モデルはゲームごとにatomic保存の `latest.pt`、引数・seed・環境・ソースhashは `config.json`（実行履歴は `config-history.jsonl`）、損失・得点・時間は追記 `train.jsonl`。再開時はモデル・optimizer・RNG・累積episodeを復元します。phase境界は `--imitation-games` の累積episode数なので再開時も同じ値を指定してください。同じ出力先で新規学習を始めるとcheckpointを上書きするため別名を推奨します。checkpointは信頼するローカルファイルのみ読み込んでください。

## ルールと設計

[Mattel公式ルール](https://service.mattel.com/instruction_sheets/FWG43-Eng.pdf)を確認。14×14、1〜5マスのfree polyomino全21種89マス。D4回転反転を正規化し91 orientation、全盤上配置を事前生成。初手は指定点を覆い、以後同色と角接触し同色の辺に接触禁止。相手との辺・角接触は自由。置けない場合のみpassし、連続2passで終了。残駒マス数を負点、全部置いたら15点、最後がmonominoなら20点。

`engine.py` の列挙はbitboard＋接触可能な角の索引。`is_legal` は座標集合による別実装で、passも全配置を独立検査。対局の通常手はこのcheckerを通します。AI内部で列挙結果が空のときのpassは高速経路を使用。駒IDはサイズ順・正規化座標辞書順で、pentominoの慣用名ではありません。

`ai.py` のheuristicは駒サイズ、新しい角、相手の角の占有、中央・遠方への進出などを加重評価。neuralは候補12特徴と状態46特徴を結合したMLP policy、および状態value。全合法候補をGPUで一括採点し、切り捨てはありません。valueは学習baselineに使い、対局時はpolicyを使います。

初期フェーズはheuristic全候補のsoftmax分布に対するcross entropy＋中心化logitの二乗誤差による**模倣学習**（90%教師手、10%ランダム）。後半は両者が現policyからsamplingする**自己対戦actor-critic**で、終局得点差/109を報酬、value回帰とentropy正則化を使用。ゲーム単位で勾配を平均しclip、Adamで更新。模倣lrは既定1e-3、自己対戦lrは既定1e-4（`--selfplay-lr`）。再開時もこれらCLI設定を適用します。自己対戦で改善できる経路がありますが、短時間の実行だけで改善や強さは保証しません。

## 限界・検証

CNNによる盤面理解・MCTS・多プロセス対局は未実装。特徴が盤面の全情報を持たず、長期的な閉塞や駒形状の理解に限界があります。CPUの手生成・特徴抽出が推論時の主な律速で、小規模MLPは5090の能力を使い切りません。短期学習・少数評価の成績を強さの証明とは扱いません。固定seedの同一環境での再現を意図し、異なるGPU/ライブラリ間のbit一致は保証しません。

テストは種類数/全89マス/91 orientation、接触ルール、重複使用・盤外・重なり・初手点、pass/終局/得点、複数局面で全盤配置の独立checkerとの完全一致、ランダム対局保存則、勾配とcheckpoint/RNG復元を検証。実測結果・コマンドは `PROGRESS.md` と `artifacts/` を参照してください。

## 初版の実測結果（2026-11-05）

ルール検証10テストPASS。秘書の独立oracleでも40対局1208手・56局面の全合法手集合が一致しPASS。エンジンsha256は `973b1559286948bd78367512635755a1d787362b7cec81083631b4e9fcc4e6d9`。別processのCUDA連続4局と2局+保存再開2局で重み/Python/Torch/CUDA RNGがbit一致（`artifacts/secretary-review/`）。

実GPUはtorch2.7.1+cu128 / CUDA12.8 / RTX5090 sm_120。forward/backward成功、全勾配finite、GPU検証時の最大PyTorch割当73.1MB。GPUチェックは `artifacts/gpu.json`。

初回は模倣60+自己対戦140局=13.315秒。改良版は全候補教師分布＋logit回帰、模倣2000+自己対戦2000局=267.589秒。主要学習合計280.904秒（約4分41秒）、再開smoke等を含めても10分未満。学習ゲームは合計4200局。追加学習の内訳はCPU手生成41.81秒、特徴60.38秒、推論/選択33.50秒、残りは主にbackward/update・保存等。

選択用seed1000〜1009、対heuristic20局での結果:

| モデル | 勝/敗/分 | 平均得点差 |
|---|---:|---:|
| 未学習seed42 | 10/10/0 | +2.35 |
| 初回模倣60局 | 4/16/0 | -5.20 |
| 初回自己対戦後200局 | 10/10/0 | -2.20 |
| 改良模倣2000局 | 2/10/8 | -1.20 |
| 改良自己対戦後4000局 | 10/10/0 | -0.20 |

勝+0.5×引分、同率なら得点差で選択。学習済み最良は改良4000局モデル `artifacts/best-trained.pt`、全体最良は**未学習**の `artifacts/best-evaluated.pt`。`selection.json` に出典を明記。全段階のcheckpointを保全しています。

その後、モデル固定のまま未使用seed2000〜2049で相手ごと100局（先後50局ずつ）を評価。最終結果を使った再選択はしていません。

| 評価対象 | 対random 勝/敗/分・平均得点差 | 対heuristic 勝/敗/分・平均得点差 |
|---|---|---|
| heuristic | 96/1/3・+25.30 | 16/16/68・0.00 |
| 未学習seed42 | 99/1/0・+44.15 | 50/50/0・+2.39 |
| 学習済み最良4000局 | 98/2/0・+19.44 | 50/50/0・-1.42 |

**自己対戦は実行できていますが、random/heuristic相手の評価では未学習からの強さの改善は示せませんでした。** 未学習も入力に手工芸特徴を使うため、一様ランダムAIとは異なります。対heuristicは両者ほぼ決定的なのでseedを変えても同じ展開が多く、100局を独立した100種類の戦略検証とは解釈できません。勝率の有意差を主張しません。

最終学習モデル評価200対局で7272手・1,693,351候補、CPU手生成2.343秒（0.322ms/手）、特徴抽出2.936秒、GPU推論＋転送0.771秒。CPU両処理合計は推論の約6.8倍で推論時の律速です。全体wall timeやtraining backwardとは別計測であり、GPUのみのkernel時間ではありません。

```bash
# 推奨の学習済みモデルで人間対AI
.venv/bin/python -m blokus play --checkpoint artifacts/best-trained.pt
# 今回の追加学習を再現するコマンド（別出力先へ）
.venv/bin/python -m blokus train --games 4000 --seconds 360 --imitation-games 2000 --selfplay-lr .0001 --seed 42 --output artifacts/reproduce
# 最終評価の再現
.venv/bin/python -m blokus evaluate --checkpoint artifacts/best-trained.pt --games 100 --seed 2000 --output artifacts/recheck.json
```

残課題は盤面全体入力・より良い報酬/探索・複数seed学習・対戦相手プール・多様な開始局面による統計評価・CPU手生成の並列化・GUIです。現在のモデルが強いBlokusプレイヤーであるとは主張しません。


### 秘書による独立直接対戦（モデル固定後の追加検証）

4000局学習済み vs 未学習の直接対戦を、未使用seed3000〜3049・先後交替100局で実施。**学習済み56勝44敗0分、平均得点差+2.90**。`artifacts/secretary-review/head_to_head.json` のcheckpoint hashを実ファイルと照合済み。この結果はモデル再選択に使用していません。直接対戦では小幅な優位が観測されましたが、random/heuristic相手では未学習より劣る指標もあり、安定した強化成功とは断定できません。

推奨する学習済みcheckpointは `/home/eletim/blokus-duo-ai/artifacts/best-trained.pt`（改良4000局、模倣2000＋自己対戦2000）。学習済みを使う用途ではこちらを指定してください。`best-evaluated.pt` は開発用基準相手で選択した未学習モデルで、名前だけで学習済みと誤認しないよう `selection.json` と合わせて参照してください。
