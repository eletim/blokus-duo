# Blokus Duo AI

Blokus Duoのルールエンジン、CLI対戦、3系統のGPU学習・評価、およびCPUによる評価重み最適化を実装しています。開発・学習環境はE-frontierの `/home/eletim/blokus-duo-ai`。

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

## 現在のモデルで対戦

推奨は既存の終局価値モデル（pooled seed201、alpha0.5、gate24）です。表示される合法手の番号を入力します。

```bash
.venv/bin/python -m blokus.value_experiment play --checkpoint artifacts/value-search/compact/seed-201/predictive-best.pt --alpha .5 --gate 24 --scheduler incremental --budget-ms 6
```

共通6ms予算・fresh96prefixの実験では非学習探索への改善が観測されました。ただし実消費時間の完全一致ではなく、相手poolと開始局面を限定した結果です。最強モデルとの断定や任意の条件への保証はしません。

追加のpaired-rankingは現valueを上回る改善を示せず、実験用のまま保持しています。[最終比較と限界](artifacts/paired-ranking/conclusions.md)、[再現手順](artifacts/paired-ranking/METHOD.md)を参照してください。全38テストと工房側のCUDA保存再開・ラベル再生監査を通過しています。下記の旧版手順・過去の推奨は各実験時点の記録です。

## 初版（v1）の対局

```bash
# 人間が先手X、学習AIが後手O
.venv/bin/python -m blokus play --checkpoint artifacts/run/latest.pt
# AI同士（CPUでも可）
.venv/bin/python -m blokus play --players heuristic random --device cpu --seed 7
# 学習AI対heuristic
.venv/bin/python -m blokus play --players neural heuristic --checkpoint artifacts/run/latest.pt
```

盤面は行・列とも0始まり。人間は表示される合法手一覧のindexを入力します。各手にはpiece/orientation/row/colと配置セルを表示。合法手なしなら自動pass。開始点はX=(4,4)、O=(9,9)。CLIは大量の合法手をテキスト表示するためGUIほど快適ではありません。終了はCtrl-C。

## 初版（v1）の学習と評価

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

## 共通ルールと初版（v1）の設計

[Mattel公式ルール](https://service.mattel.com/instruction_sheets/FWG43-Eng.pdf)を確認。14×14、1〜5マスのfree polyomino全21種89マス。D4回転反転を正規化し91 orientation、全盤上配置を事前生成。初手は指定点を覆い、以後同色と角接触し同色の辺に接触禁止。相手との辺・角接触は自由。置けない場合のみpassし、連続2passで終了。残駒マス数を負点、全部置いたら15点、最後がmonominoなら20点。

`engine.py` の列挙はbitboard＋接触可能な角の索引。`is_legal` は座標集合による別実装で、passも全配置を独立検査。対局の通常手はこのcheckerを通します。AI内部で列挙結果が空のときのpassは高速経路を使用。駒IDはサイズ順・正規化座標辞書順で、pentominoの慣用名ではありません。

`ai.py` のheuristicは駒サイズ、新しい角、相手の角の占有、中央・遠方への進出などを加重評価。neuralは候補12特徴と状態46特徴を結合したMLP policy、および状態value。全合法候補をGPUで一括採点し、切り捨てはありません。valueは学習baselineに使い、対局時はpolicyを使います。

初期フェーズはheuristic全候補のsoftmax分布に対するcross entropy＋中心化logitの二乗誤差による**模倣学習**（90%教師手、10%ランダム）。後半は両者が現policyからsamplingする**自己対戦actor-critic**で、終局得点差/109を報酬、value回帰とentropy正則化を使用。ゲーム単位で勾配を平均しclip、Adamで更新。模倣lrは既定1e-3、自己対戦lrは既定1e-4（`--selfplay-lr`）。再開時もこれらCLI設定を適用します。自己対戦で改善できる経路がありますが、短時間の実行だけで改善や強さは保証しません。

## 初版（v1）の限界・検証

初版（v1）はCNN・探索を使わない構成です。現在の研究実装はCNNと幅制限2手探索に対応していますが、MCTS・多プロセス対局は未実装です。初版の特徴は盤面の全情報を持たず、長期的な閉塞や駒形状の理解に限界があります。CPUの手生成・特徴抽出が推論時の主な律速で、小規模MLPは5090の能力を使い切りません。短期学習・少数評価の成績を強さの証明とは扱いません。固定seedの同一環境での再現を意図し、異なるGPU/ライブラリ間のbit一致は保証しません。

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

# 3方式研究 round1（research/three-methods）

v1 CLIとcheckpointは維持し、研究機能は `python -m blokus.experiment` に分離しました。エンジンは秘書検証済みのfrontier bitboard高速化のみ取り込み、ルール・合法手順序を保持しています。

## 方式・比較の読み方

- **A（軽量候補MLP）**: 候補12特徴＋研究state49特徴。固定heuristic priorに学習する残差を加算。v1とheuristicを組み合わせた全候補教師分布を蒸留し、小さなon-policy actor-critic補助を併用。相手はrandom/heuristic/v1/current-selfから一様選択。
- **B（全盤面CNN）**: 自石・相手石・双方の開始点の4×14×14を2層CNNへ入力。全盤面flattenと候補配置セルの局所conv特徴、残駒等49特徴をpolicy/valueへ入力。教師・相手poolはAと共通。
- **C（探索教師蒸留）**: Bと同じ構造ですが別seed・独立checkpoint。student policyでroot上位4候補を選び、各候補にheuristic上位3応手を2ply negamaxで調べる。最大16子node/決定、全候補最適の保証はありません。探索leafは手番基準の得点差＋frontier差、終局は正確な得点差。探索済み候補だけで教師分布を作り、CNNを蒸留します。student更新で次のshortlistが変わる簡略expert iterationです。Cの教師行動はoff-policyなのでactor-critic補助を適用しません。

研究stateはv1の46特徴に手番側score/89、相手score/89、passes/2を加え、全置き15/20点差を保持。v1互換は維持しています。Aは盤面全体を見ないため状態の情報落ちは残ります。全方式でvalueは終局得点差/109を回帰。相手手番では符号が反転します。

A/B/Cの未学習policyは同じ明示的なheuristic priorです（学習残差の最終層はゼロ初期化）。「未学習」を一様randomと同一視しません。今回のA/Bは安定した教師蒸留が主、自己対戦強化学習のみの比較ではありません。C-offは学習寄与、C-searchは同じ学習済み重みに推論時探索を追加した寄与を測ります。探索条件の思考時間は増えるため、時間同等比較とは主張しません。

## 実行・再開・対戦

```bash
cd /home/eletim/blokus-duo-ai
# 共通180秒/方式。新規の出力ディレクトリを指定
.venv/bin/python -m blokus.experiment run --seconds 180 --validation-prefixes 12 --holdout-prefixes 24 --output artifacts/research-reproduce
# 個別学習。Cも必ず探索教師から独立checkpointを学習
.venv/bin/python -m blokus.experiment train --method C --seconds 180 --output artifacts/new-C
# 保存再開：optimizer/RNG/累積episodeを復元、secondsは今回追加分
.venv/bin/python -m blokus.experiment train --method C --resume artifacts/new-C/latest.pt --seconds 60 --output artifacts/new-C-resumed
# 学習済みCNNと人間（Xが人間）
.venv/bin/python -m blokus.experiment play --agents human artifacts/research-round1/B/selected.pt
# 探索ありC vs A
.venv/bin/python -m blokus.experiment play --agents artifacts/research-round1/C/selected.pt:search artifacts/research-round1/A/selected.pt
# 保存した同一prefix集合で総当たり。結果は新しいJSONへ
.venv/bin/python -m blokus.experiment tournament --prefix-file artifacts/research-round1/holdout-prefixes.json --agents artifacts/research-round1/A/selected.pt artifacts/research-round1/B/selected.pt artifacts/research-round1/C/selected.pt --output artifacts/research-recheck.json
.venv/bin/python -m pytest -q
```

`run` は既存の非空出力先を拒否。`train` も既存latest.ptをresumeなしで上書きしません。学習はゲーム境界で停止するため180秒を最大1対局分超過し得ます。中間と最終は別保存し、選択済み重みはselected.pt。新方式checkpointはformat2で、v1のcheckpointとは識別されます。信頼するローカルファイルのみ読み込んでください。

## prefixと不確実性

合法ランダム2〜6plyを共通prefixとして保存。validationはseed41000の12prefix、holdoutはseed51000の24prefixで分割し、集合と各prefixをSHA256で識別します。全相手で同じprefix集合を使い、各prefixの盤面を固定してAgentの担当色を交換する2局を実施。tie-break RNGは色に割り当て、相手変更でも同じseed規則を使います。

各方式で中間/最終からvalidation上のheuristic/v1への平均得点差の平均が高い方を選択（同値は中間）。`selection.json` を保存してからholdoutを開始し、その結果でcheckpointを選び直しません。holdoutはA/B/C対baseline、3方式総当たり、各方式の学習前後、C-search追加寄与を比較。baselineのv1重みとheuristic/random仕様も保存しています。

各prefix内の2局の平均を単位に4,000回bootstrapし、平均得点差と勝+0.5×引分率の95%区間を報告します。各局を独立標本として水増ししません。24prefixかつ各手法1学習seedなので**暫定評価**です。この区間は開始局面の不確実性であり、再学習seed間のばらつきや多数比較を補正した勝者確率ではありません。

## 成果物と計測

`artifacts/research-round1/` にrun-config、baseline-config、prefix、各手法config/JSONL/summary/checkpoint、validation-results、selection、holdout-results、run-summaryを保存。configとcheckpoint metadataにseed、全引数、source hash、torch/CUDA、teacher、相手poolを記録しています。学習時間は対局生成・探索教師・更新・保存を含み、model/環境初期化は別です。CUDAは逐次使用し、無期限runを残しません。

評価JSONには各手法の全decision時間、合法手生成、特徴抽出、network（転送・同期を含む）、探索秒、nodes、ms/手を分離記録。学習JSONには対局数・更新数・learner局面数・探索node数・rollout/update秒。総合性能はCNN/探索だけでなくCPU生成を含めて判断してください。

理論上の参考は [Expert Iteration](https://arxiv.org/abs/1705.08439) と [RLの不確実性評価](https://arxiv.org/abs/2108.13264)。論文のMCTSや全評価手法を再現するものではありません。

## Round1結果

実験runは582.76秒（約9分43秒、実装を除く）。各180秒の学習でA=4,104局、B=3,325局、C=2,205局を完走、Cの教師探索は457,156node。全方式で中間より最終をvalidationで選択し、その後は固定しました。全21組、計1,008局のholdoutと独立監査が完了しています。

| 条件 | 対heuristic 勝/敗/分 | 得点差 [prefix bootstrap95%区間] | 対v1 勝/敗/分 | 全体ms/手 |
|---|---:|---:|---:|---:|
| A | 26/19/3 | +0.81 [-1.12,+2.67] | 27/20/1 | 0.881 |
| B | 25/20/3 | +0.44 [-1.50,+2.33] | 24/21/3 | 0.989 |
| C探索なし | 26/19/3 | +1.29 [-1.52,+4.35] | 27/21/0 | 0.963 |
| C＋探索 | 32/14/2 | +3.00 [+0.77,+5.46] | 31/16/1 | 3.262 |

探索なし総当たりの観測point率順位はA>B>Cですが、**明確な勝者・学習改善の断定はできません**。C＋探索は今回のbaseline相手に有望ですが、同じC重みへの探索追加の直接比較は27勝19敗2分、得点差+1.19の区間[-1.08,+3.19]で0を含みます。探索追加の時間も約3.4倍です。24prefix・1学習seed・多数比較未補正という限界があります。

全表・思考時間内訳・prefix hashは [実測レポート](artifacts/research-round1/report.md)、対局ごとの値は [holdout-results.json](artifacts/research-round1/holdout-results.json)。各selected.ptは保持し、holdoutから新しいbest checkpointを選び直していません。validation criterion最大はB、holdoutでの探索ありは別条件として扱います。

```bash
# validationで選択済みのBで対戦
.venv/bin/python -m blokus.experiment play --agents human artifacts/research-round1/B/selected.pt
# 探索ありCを試す（追加思考時間を使用）
.venv/bin/python -m blokus.experiment play --agents human artifacts/research-round1/C/selected.pt:search
# 秘書作成の独立監査を再実行
.venv/bin/python scripts/audit_research.py artifacts/research-round1/holdout-results.json artifacts/research-round1/holdout-prefixes.json
# A/B/Cの実CUDA勾配と別process保存再開チェック
.venv/bin/python scripts/check_research_cuda.py artifacts/research-gpu-recheck.json
```

実験時のsource snapshotも `artifacts/research-round1/source/blokus/` に凍結しました。その後の変更はCLI出力defaultの分離と再開時の累積時間表示修正のみで、学習・評価アルゴリズムは同じです。CLIのtournament/prefixesはそれぞれ専用JSONの既定出力を持ちます。再開時のlearning rateは保存したoptimizer設定を引き継ぎます。

以下の追加改善では、教師蒸留を超える信号としてA2 CEM/進化戦略を実施しました。CPU対戦ベースの学習と明示し、学習用の開始局面と新しいholdoutを用意し、validationは既存のセットを使いました。今回のholdoutを見て設計を変えた後に同じholdoutで選択・性能主張を続けません。

# 追加改善 A2：CPU CEM

教師蒸留を超える信号を得るため、候補評価重みを実際のpaired対局の得点差から直接最適化します。**GPU勾配学習ではありません**。1世代10候補を共通の3prefixでheuristic/v1相手に先後交換し、120局の結果で上位3候補を選び、平均/標準偏差を平滑更新します。世代ごとにtrain prefixを更新し、incumbentと中心を候補に含めます。

サイズ係数12を固定して正のスケール不定性を除き、手の順位に影響する9係数を最適化。局面内で定数の2特徴は0に固定。探索標準偏差下限0.25、係数[-20,20]。学習checkpointはJSON format3（weightsとmetadata）、再開にはmean/std/incumbent/RNGを含むcem-state.jsonを使用。

学習240秒目安、最大100世代。次世代時間を直前実測×1.25で見積もり、残予算が不足すれば開始しません。世代は途中中断せず、最初の世代は必ず実測します。これは世代境界の予算制御で、OS負荷による超過が絶対にないというwall-time保証ではありません。

```bash
# 新しい出力先で追加実験（CPU学習→validation選択→新holdout）
.venv/bin/python -m blokus.cem run --seconds 240 --population 10 --train-prefixes 3 --output artifacts/cem-reproduce
# CPU学習だけ / 保存再開（generationsとsecondsは追加分）
.venv/bin/python -m blokus.cem train --seconds 240 --output artifacts/cem-train
.venv/bin/python -m blokus.cem train --resume artifacts/cem-train/cem-state.json --seconds 60 --output artifacts/cem-resumed
# 学習済みA2と対戦
.venv/bin/python -m blokus.experiment play --agents human artifacts/research-a2/selected.json
```

最終比較は新holdout seed61000の24prefixを事前固定して使います。学習seed71000+世代、validationは既存41000で、旧holdout51000は選択に使いません。A2/heuristic/v1/B（第一ラウンドvalidation最大）/C-search総当たりとA2学習前後を保存します。全候補の重み・fitness・paired対局結果・世代prefixは `artifacts/research-a2/train/` にあり、結果が良かった世代だけを抜き出して学習成績と見せることはしません。

秘書の独立監査で第一ラウンド24unique開始状態・21組1008局・選択基準・hash・時間集計がPASS。またA/B/Cすべてで別processのCUDA連続4局と2+保存再開2局のmodel/Python/Torch/CUDA RNGがbit一致。資料は `artifacts/secretary-review/round1-audit.json` と `research-cuda-resume/result.json`。実装側でもoptimizer/NumPy RNGを含めたCUDA再開と有限勾配を検証し、`artifacts/research-round1/gpu-verification.json` に保存しました。

A2-searchの比較も新holdoutの結果を見る前に宣言しました。A2-offのvalidation選択重みをそのままroot候補絞り込みに使い、固定depth2・幅4×3で探索します。A2-off/C-search/heuristic/v1との4組を `search-results.json` に別保存します。

```bash
.venv/bin/python scripts/evaluate_cem_search.py artifacts/research-a2
.venv/bin/python -m blokus.experiment play --agents human artifacts/research-a2/selected.json:search
```

時間上限の学習では実行速度により局数・世代数が変わります。重みの厳密な再現には各summary/checkpointに記録された局数・世代数を指定し、secondsには十分な上限を与えてください（A/B/Cは`--games`、A2は`--generations`）。seed・固定baseline・同じコードと依存環境も必要です。実験時のコードを各成果物のsource/に凍結しています。

## A2追加実験の結果

CPU学習は237.89秒、96世代、11,520対局、960候補評価で停止。validationで90世代目を選択し、重みを固定しました。学習prefix288個とvalidation/新旧holdoutの重複は0。新holdoutでの通常比較11組と追加探索比較4組はどちらも独立監査スクリプトPASSです。

| 条件（左 vs 右） | 勝/敗/分 | 平均得点差 [prefix bootstrap95%区間] |
|---|---:|---:|
| A2 vs heuristic | 26/18/4 | +2.83 [+0.58,+5.21] |
| A2 vs v1 | 28/17/3 | +2.46 [+0.60,+4.33] |
| A2 vs B | 29/16/3 | +2.92 [+1.21,+4.73] |
| A2 vs C-search | 21/24/3 | -3.58 [-6.04,-1.25] |
| A2-search vs A2 | 24/23/1 | +1.19 [-1.33,+3.81] |
| A2-search vs C-search | 28/19/1 | +1.71 [-0.46,+4.04] |
| A2-search vs heuristic | 33/13/2 | +2.83 [+0.96,+4.62] |
| A2-search vs v1 | 29/16/3 | +2.44 [+0.71,+4.15] |

A2-offは今回の新holdoutでheuristic/v1/Bへの平均得点差に改善が観測されました。ただし勝率の区間では0.5を含む組もあり、単一学習seed・24prefix・多数比較未補正です。A2-searchとC-searchの優劣は未確定です。第一ラウンドとprefixが異なるため、ラウンド間の数字をそのまま改善量として比較しません。

推論全体はA2-off 0.545ms/手、A2-search 2.967ms/手、C-search 3.370ms/手。CPU線形モデルが軽量な一方、探索には追加時間が必要です。詳しくは [A2実測レポート](artifacts/research-a2/report.md)、元データは同ディレクトリの `holdout-results.json` / `search-results.json` を参照。

現時点の対戦用CLI（低負荷ならoff、探索を試す場合はsearch）:

```bash
.venv/bin/python -m blokus.experiment play --device cpu --agents human artifacts/research-a2/selected.json
.venv/bin/python -m blokus.experiment play --device cpu --agents human artifacts/research-a2/selected.json:search
```

全27テストPASS。A2-searchとの人間入力経路も入力indexを与えて終局まで確認。追加A2の学習/評価は終了し、無期限runを残していません。秘書の追加「学習なしheuristic-search」対照診断はcheckpoint選択と切り離して記録します。

## 秘書の探索対照診断（seed81000、選択には不使用）

学習なし `heuristic:search` と同じ幅4×3で独立比較した結果、C-searchは27勝19敗2分・平均+1.50（95%区間[-0.67,+3.65]）、A2-searchは16勝28敗4分・平均-0.54（[-2.75,+1.58]）。A2-search vs C-searchも24勝22敗2分・+1.27（[-0.65,+3.10]）で優劣未確定です。

学習なしheuristic-search自体がheuristicに30勝15敗3分でした。したがって**探索ありの好成績を、学習した候補順位の効果だけとは解釈できません**。低負荷のA2-offは今回の新holdoutで改善が観測された一方、探索ありの明確な勝者は不明です。全checkpointは固定したまま。診断データとPASS監査は `artifacts/secretary-review/search-controls/` に保存しています。

```bash
# 学習なし・同じ幅4×3の探索対照も試せます
.venv/bin/python -m blokus.experiment play --device cpu --agents human heuristic:search
```

## 終局結果を学ぶ価値探索（追加研究）

教師の候補順位ではなく、実際に終局まで進めた得点差を学ぶCNNを追加しました。結果と全比較は [価値探索レポート](artifacts/value-search/report.md)、計画・失敗と修正は [PROGRESS.md](PROGRESS.md) に保存しています。既存エンジンと既存モデルは変更していません。

入力は手番基準の6面盤面tensorと残駒・得点・pass等49特徴。静的評価にCNNの補正を加え、2ply探索の葉を評価します。終局はネットの予測を使わず正確な得点を返します。学習ラベルはepsilon付きheuristic/非学習探索/A2探索の完走結果で、反実仮想の探索葉にも独立した完走ラベルを作ります。これは終局結果の教師あり回帰であり、最適対戦価値やオンライン自己対戦強化学習が完成したという意味ではありません。

初回の過学習に対し、独立した探索葉の追加、終盤限定の価値使用、モデル小型化、盤面対称変換、6,000局の追加、複数rootのGPU一括推論まで3段階を実測しました。各段階3seed。validationはseed91000、段階ごとのfresh holdoutは92000/93000/94000です。holdoutからcheckpointを選び直しません。予測検証も同一対局のフレームをtrain側と混ぜません。

時間比較は全方式に**合法手生成・順位付け・GPU同期込み6ms/手の共通予算**を与えています。C対照は既存Cの候補順位を共通探索器で使います。元の固定幅C-searchそのものと同一アルゴリズムではありません。OS/native演算を強制停止できないため厳密なリアルタイム上限ではなく、実時間・超過時間・node数を記録し、遅れて終了した子/バッチの評価を採用しません。実時間をpaddingで揃えた比較ではありません。時間による枝数変化があるため、対戦の完全bit再現は保証しません。学習の重み/optimizer/RNG再開は別途検証します。

```bash
# 全体の第1・第2段階（新しい出力先で実行）
.venv/bin/python -m blokus.value_experiment run --output artifacts/my-value-run
# 小型モデルの第3段階（既定の第1・第2段階データが必要。既存結果を上書きしない）
.venv/bin/python scripts/run_value_compact.py
# 任意の学習・再開。stepsは今回追加する更新数
.venv/bin/python -m blokus.value_experiment train --data artifacts/value-search/compact/expanded.npz --output artifacts/my-value-model --arch pooled --augment --steps 4000 --seconds 120 --seed 201
.venv/bin/python -m blokus.value_experiment train --data artifacts/value-search/compact/expanded.npz --output artifacts/my-value-model --arch pooled --augment --steps 1000 --seconds 120 --seed 201 --resume artifacts/my-value-model/latest.pt
# 保存済みprefix/specsで対戦。specsは名前→BudgetAgent引数のJSON
.venv/bin/python -m blokus.value_experiment tournament --prefix-file artifacts/value-search/compact/holdout-prefixes.json --specs artifacts/value-search/compact/play-specs.json --output artifacts/my-tournament.json --budget-ms 6
# 結果の算術・時間・hash・prefix監査とseed×prefix CI再計算
.venv/bin/python scripts/analyze_value.py artifacts/value-search
.venv/bin/python scripts/check_value_resume.py
.venv/bin/python -m pytest -q
```

`config.json`とcheckpointにseed/torch/source hash/データhash/更新数を保存します。`latest.pt`は再開用、`predictive-best.pt`は未使用対局の予測誤差による候補、`selected.pt`と`selection.json`は対戦validationで混合率を確定したものです。GPUデータ生成ではなくCPU6process生成＋GPU minibatch学習です。探索候補の分布、継続方策と真の対戦相手の違い、得点回帰誤差と候補間順位誤差の違いが残る課題です。

### 最新の確認結果（第4段階）

一括推論版は時間超過が多く推奨しません。小型モデルを逐次探索へ戻し、validationで混合率を固定してfresh holdout seed95000で再確認しました。3seed平均得点差は非学習探索に **+2.15 [95%区間 +0.29,+3.88]**、共通時間制御版Cに **+1.19 [-0.50,+2.90]**。学習による改善の兆候はありますが、安定した優越は未確定です。詳細は [結論・限界・次案](artifacts/value-search/conclusions.md)。最終モデルは実測4.58〜4.69ms/手、共通予算6ms、実時間の完全一致ではありません。

試用にはvalidation最高criterionのseed201を使います（holdoutの最高seedを選んだものではありません）。

```bash
.venv/bin/python -m blokus.value_experiment play --checkpoint artifacts/value-search/compact/seed-201/predictive-best.pt --alpha .5 --gate 24 --scheduler incremental --budget-ms 6
# 第4段階の再実験スクリプト（既存結果があれば保全して停止）
.venv/bin/python scripts/run_value_confirm.py
# 最終比較のspecs
.venv/bin/python -m blokus.value_experiment tournament --prefix-file artifacts/value-search/confirm/holdout-prefixes.json --specs artifacts/value-search/confirm/play-specs.json --output artifacts/confirm-replay.json --budget-ms 6
```

GPU学習は9モデル計24,000更新・約30.67秒（smoke/再開検証別）。4段階のholdout計1,344局、validation計1,728局。全34テスト、別process CUDA再開、開始状態分離、結果算術監査とCLI終局確認はPASS。追加学習は稼働していません。

## 同一局面候補の差分・ランキング学習

終局価値モデルから初期化し、同じrootの4候補それぞれを共通の継続seedで終局まで進めた得点差を学ぶ経路を追加しました。損失は候補間差分回帰＋soft順位＋弱い絶対値回帰です。CPU6workerで1,200対局から2,400局面を取り、19,200候補継続を生成。GPU学習seed301/302/303、新validation32prefixで選択し、fresh holdout96prefix×先後交換で非学習探索・現valueモデルを比較しています。

初回の時間ログから、一度の大遅延により探索費用推定が高止まりして探索が再開しなくなる不具合を発見し修正しました。旧ログを保全し、モデル・混合率は変更せず新holdout230000/96prefixで全組を確認しています。最終結果は [確認版レポート](artifacts/paired-ranking/confirmed/report.md)、設計・完全な再現コマンドは [METHOD.md](artifacts/paired-ranking/METHOD.md) を参照してください。共通6ms予算であり、実消費時間をpaddingで一致させた比較ではありません。

```bash
# 新規出力先で生成・3seed学習・validation・初回holdoutまで
.venv/bin/python -m blokus.ranking_experiment run --output artifacts/my-ranking
# 固定モデル、時間制御修正後の最終prefixを使った再比較
.venv/bin/python -m blokus.value_experiment tournament --prefix-file artifacts/paired-ranking/confirmed/holdout-prefixes.json --specs artifacts/paired-ranking/play-specs.json --output artifacts/ranking-replay.json --budget-ms 6
# 差分モデルを試す（seed301はvalidation最大、holdoutでは選び直していない）
.venv/bin/python -m blokus.value_experiment play --checkpoint artifacts/paired-ranking/seed-301/selected.pt --alpha .5 --gate 24 --budget-ms 6
```

最終96prefix確認では3seed平均得点差が非学習探索に **+2.58 [95%区間 +1.77,+3.32]**、現valueに **−0.07 [−0.69,+0.58]**。今回の追加学習による置換の根拠は得られなかったため、推奨は既存valueのままです。主学習18,000更新/27.16秒、全38テスト・CUDA別process再開・256終局ラベル再生監査PASS。[原因分析・次案・全計算量](artifacts/paired-ranking/conclusions.md)を保存しています。

## 実探索に合わせたデータと損失比較（実験）

相手上位3応手を持つgroupと、既存valueを含む継続poolを追加しました。pilotでラベルの安定性を調べて8完走を採用し、共通データ・初期checkpoint・各6000更新で絶対値のみ/差分併用/順位併用を各3seed比較。新validationで順位併用を固定し、その3seedだけをfresh96prefixで評価しました。

非学習探索に平均 **+3.11 [95%区間 +2.34,+3.92]**、既存valueに **+0.28 [−0.31,+0.88]**。追加改善は未確認なので**推奨は既存valueのまま**です。共通6ms予算で、実消費時間の完全一致ではありません。

[結果・計算量・原因と次案](artifacts/search-aligned/conclusions.md)／[再現手順と乱数共有の定義](artifacts/search-aligned/METHOD.md)。主学習9モデル54,000更新/105.88秒、全41テストと3損失の別process CUDA保存再開PASS。新実験はresearch/search-alignedで保持しています。

```bash
# 実験モデルの試用例（代表seed。既存推奨の置換ではありません）
.venv/bin/python -m blokus.value_experiment play --checkpoint artifacts/search-aligned/ranking/seed-501/selected.pt --alpha .5 --gate 24 --budget-ms 6
```

## 実戦局面の多様性・checkpoint選択（実験）

実6ms探索が訪れた局面から、**9,256root×4完走**と**2,314root×16完走**を生成しました。CPU生成は967秒対976秒、継続ply差は約0.49%。絶対値損失を固定して各3更新seedを学習し、raw診断と実戦alpha=.5・終局上書きを含む意思決定損失によるcheckpoint選択を比較しています。

新validationでは少数root・16完走のraw選択を固定。fresh96prefix・先後交換の最終評価では、3seed総合で非学習探索に **+2.35 [95%区間 +1.54,+3.20]**、既存valueに **−0.10 [−0.78,+0.48]**。今回の追加改善は確認できず、**既存value推奨を維持**します。共通6ms予算ですが実消費時間の完全一致ではありません。3seedはデータ・初期重みを共有し、区間はseed/prefix交差bootstrapです。

GPU主学習は54,000更新・87.47秒。全44テスト、別process CUDA保存再開、ラベル標本再生、全評価着手・得点・CI・時間・checkpoint選択監査はPASS。学習root/leafと収集ゲーム中終盤のhash重複も検査しました。全継続内部局面・類似局面の非重複を保証するものではありません。

固定重みの6/25/100ms曲線は8開発prefixの小規模診断で、長時間側の得点差は正でしたが全CIが0を含みます。学習改善と探索時間増加を混同しません。root数4倍でも元ゲーム数は815対780で、独立展開の多様性が4倍になった比較ではない点も限界です。

[詳細結果・原因・次案](artifacts/visited-diversity/conclusions.md)／[設計と再現手順](artifacts/visited-diversity/METHOD.md)。

```bash
# 固定した最終7対戦を再実行（再選択には使わない）
.venv/bin/python scripts/replay_visited.py --output artifacts/visited-replay.json
# 研究用モデルを試用。seed601は代表例でありholdoutによるseed再選択ではない
.venv/bin/python -m blokus.value_experiment play --checkpoint artifacts/visited-diversity/full/dense/seed-601/raw.pt --alpha .5 --gate 24 --budget-ms 6
# 検証一式。独立checkerの全手再生を含むため約10分
.venv/bin/python scripts/verify_visited.py
```

## 固定重みの反復深化・終盤厳密探索（実験）

追加学習を止め、既存valueを固定して探索器を比較しました。候補幅を制限する反復深化と、全合法手を使う低mobility終盤の厳密探索を実装。TT bound、pass・ボーナス・視点、中断時の完了反復fallbackを検証しています。幅を絞った探索は完全探索ではありません。

新validationで25msを固定し、別々のfresh96prefixを先後交換して同予算の旧2plyと比較しました。

- 反復深化: 94勝91敗7分、得点差 **+0.104 [95%区間 −0.151,+0.339]**。
- 終盤hybrid: 90勝89敗13分、**+0.214 [−0.068,+0.510]**。厳密解96/2958手、実平均17.624ms対旧17.416ms。

安定した改善は未確認で、**既存value/2ply推奨を維持**します。学習重みの改善ではありません。共通予算は実時間の完全一致やhard deadline保証ではなく、超過率も保存しています。学習0更新、有限run合計1376局/1203秒、全53テスト・独立終盤全探索・全対局再生/算術監査PASS。

[全結果・計算量・限界と次案](artifacts/exact-hybrid/conclusions.md)／[再現手順](artifacts/exact-hybrid/METHOD.md)。研究用CLI:

```bash
.venv/bin/python -m blokus.deep_experiment play --spec artifacts/exact-hybrid/selection.json
```

## 応手脱落の固定node診断（実験）

独立16ゲームから各1中盤・1終盤を取り、root幅・応手幅・深さを共通2048node上限で切り分けました。中盤の選択4ply参照では、root平均48.2%で最悪応手がheuristic上位3の外にあります。ただし応手3→12だけの選択損失改善は0.010点に留まり、葉評価誤差と完了候補数のトレードオフが残りました。中盤参照は厳密解ではありません。

変更を応手幅だけに限定し、新validationで幅6を固定。試合ごとcold・共通25ms・fresh96prefix先後交換では旧幅3に **89勝98敗5分、−0.240点 [95%CI −1.146,+0.615]**。実時間17.726/17.541ms、平均完了root20.53/21.68。**追加改善は未確認で、既存value/2ply推奨を維持**します。以前のwarm-cache結果は変更していません。

全60テスト、全16終盤の独立全探索、固定node再現、評価再生・算術監査PASS。GPU追加学習なし。[詳細結果・原因と別系統への次案](artifacts/reply-diagnostics/conclusions.md)／[再現コマンドと指標定義](artifacts/reply-diagnostics/METHOD.md)。研究用CLIは `.venv/bin/python -m blokus.reply_experiment play`。

## 終局rolloutへの逐次配分（実験）

共通乱数で候補から終局まで試し打ちし、均等配分と16＋8＋8完走の逐次絞り込みを比較しました。25msでは標本不足だったため、開発・validationで上位8候補/逐次配分/100msを固定し、fresh96prefixを先後交換しています。

- 同100ms旧2plyに **71勝113敗8分、−3.031点 [95%CI −4.516,−1.547]**。
- 均等配分への直接対照は **101勝84敗7分、+1.125点 [+0.375,+1.891]**。

配分改善は見られましたが、方式全体は旧2plyを下回り、**既存value/2ply推奨を維持**します。主比較の実平均73.28ms対旧55.42ms、fallback15.7%。時間超過は57.2%で主比較最大100.97msのため、hard deadline保証はありません。固定相手方策の平均rolloutは、相手が最悪応手へ適応する値とは異なります。

学習0、512局1073.90秒、全68テスト・終局ラベル/全対局監査PASS。[詳細・原因と次案](artifacts/terminal-rollouts/conclusions.md)／[再現手順](artifacts/terminal-rollouts/METHOD.md)。研究用CLI: `.venv/bin/python -m blokus.rollout_experiment play`。

## 広い2plyを保持した候補優先追加探索（実験）

固定valueで旧方式と同じroot上位128/相手上位3の2plyを初期評価し、余剰時間で候補単位の深さ4 negamaxを追加しました。枝限定探索であり完全minimaxではありません。未完了の追加値は使わず初期値を保持、試合ごとcold。学習重みは変更していません。

新validationで100msを固定後、fresh96prefix・先後交換:

- 同100ms旧2plyへ **114勝68敗10分、+2.479点 [95% prefix CI +1.422,+3.620]**。
- 同一初期処理・追加なしへ **111勝70敗11分、+2.490点 [+1.401,+3.625]**。

この条件では追加探索に正の差が出ました。主比較実時間は70.679ms対51.612msで、**共通上限での改善であり実消費時間の完全一致ではありません**。追加側3001手の超過・初期fallbackは0、追加root13319完了。25msは開発中盤で追加探索できず、priority対uniformの優位も未確認です。**mainの既存value/2ply推奨は維持**し、採用は独立レビュー後に判断します。

全74テスト、480対局全再生・算術監査、小終盤12独立解照合PASS。主要実験約15.8分、追加学習0。[詳細・制約・次のpolicy学習案](artifacts/minimax-refinement/conclusions.md)／[再現手順](artifacts/minimax-refinement/METHOD.md)。研究用CLI:

```bash
.venv/bin/python -m blokus.refinement_experiment play --budget 100
```

## 教師の再確認と低予算root policy蒸留（実験）

教師の設定・重みを固定し、別のfresh96prefix・先後交換・共通100msで旧2plyへの得点差を再確認しました。**98勝89敗5分、+1.406点 [95% prefix CI +0.234,+2.625]**。同じ実装の独立標本再確認であり、独立実装監査とは呼びません。実68.552ms対49.497msで、実時間完全一致でもありません。

新規300ゲームから、同深度比較が可能な571rootを用いてroot順位policyを3更新seedで蒸留。相手応手とvalue/2plyは固定しました。新validationで25ms・1000更新を固定後、fresh96prefix・先後交換・全3seedでは:

- 旧2plyへ **+0.203点 [−0.832,+1.325]**、261勝276敗39分。
- 未学習順位モデルへ **+0.321点 [−0.821,+1.526]**、269勝270敗37分。

**学生の学習改善は未確認、既存低予算value/2ply推奨を維持**します。区間は3seed・先後をprefix内でまとめて算出。教師の成績を学生へ流用せず、holdout再選択もしていません。一致率+5ptの当初目安は未達でしたが、実戦validationの信号を根拠にfresh確認へ進めた判断変更は事前に記録しています。

dev診断では25ms時に教師手を全て評価対象へ含めても、最後の2ply値で別手を選ぶ例が多く、順位変更だけの限界が見えました。追加学習は3seed計8.05秒、全78テスト・CUDA別process再開・1856評価対局監査PASS。main統合は行っていません。[詳細結果・原因・次案](artifacts/root-policy-distillation/conclusions.md)／[設計と再現CLI](artifacts/root-policy-distillation/METHOD.md)を参照してください。

## 最終root値への補正学習（研究限定）

depth4−depth2を全盤面＋候補の小型CNNで学び、完了したrootの最終値に加える経路を追加しました。固定value本体と相手上位3応手を維持し、未知ラベルはmask、同値候補は許容集合として評価します。240ゲーム477局面、3更新seed、ゼロ補正推論対照を用意しました。

新validationで3000更新/強度0.5/25msを固定後、fresh96prefix・先後交換で評価。3seed平均の得点差は旧value/2plyへ **+0.036 [95% prefix CI −1.005,+1.031]**、同じモデルを推論するゼロ補正へ **−0.057 [−1.038,+0.912]**。学習改善は確認できず、既存低予算推奨を維持します。devでの教師選択損失は改善しましたが、別ゲームでの予測・選択誤差が改善せず、過学習と小規模validation選択の分散が課題です。[結果・原因分析・計算量](artifacts/root-value-correction/conclusions.md)、[設計・再現手順](artifacts/root-value-correction/METHOD.md)を参照してください。

```bash
# 研究用補正学生 vs 旧value/2ply（採用モデルではありません）
.venv/bin/python -m blokus.correction_experiment play --seed 1001
```

82 tests、CUDA別process再開、477局面と評価1,632局の実装側再生・算術監査PASS。GPU主学習は3seed計9,000更新/10.31秒。検証済み100ms教師はmain **cbe3b91**に追加オプションとして統合済みで、mainでは `python -m blokus.teacher_play` で起動できます。教師は共通100ms上限の2標本で正の平均得点差があり、実消費時間は旧探索より長く、勝率優越は未確定です。教師の成果を学生の改善とは扱いません。

## 独立ゲーム数による補正学習曲線（対戦へ進まず）

新しいゲームで、予定訓練240/960/2400ゲームの入れ子集合・3分割・3更新seed・正則化2条件を比較しました。固定評価は356有効ゲーム。2400条件で教師補正の平均MSEはzero7.304から6.909–6.928へ改善傾向がある一方、候補選択損失の改善は分割をまたいで再現せず、CIも0を含みました。**事前固定の対戦gate未達により、対戦validation/holdoutは生成していません。既存value/2ply推奨を維持し、mainは変更していません。**

[学習曲線と結論](artifacts/correction-game-scaling/conclusions.md) ／ [学習曲線PNG](artifacts/correction-game-scaling/learning-curves.png) ／ [設計・再現](artifacts/correction-game-scaling/METHOD.md)

3,960ユニーク元ゲーム・7,424root、54run/162,000更新、GPU主学習176.53秒。85 tests、CUDA別process再開、ラベル再生・分割hash・CI監査PASS。追加生成gateだけはMSE改善を根拠に例外を記録して2400まで測定しましたが、対戦gateは変更していません。前回の再開表記は「40連続 vs20＋追加20＝通算40」が正確です。候補入力の情報欠落も診断しましたが、観測分だけで全体の失敗を説明できるとは主張していません。

### 候補セルmask対照（研究限定）

[実験レポート](artifacts/candidate-mask/REPORT.md): 従来要約と正確な配置maskを3ゲーム分割×3更新seedで比較。配置を入力・勾配で区別できるが、固定汎化gateは未達。対戦holdoutなし。mainの既存value/2ply推奨は変更しない。
