# Blokus Duo AI 実装進捗

2026-11-05 初版開始。専用 /home/eletim/blokus-duo-ai は存在しなかったため新規作成。/、/home、/home/eletim に AGENTS.md なし。村・他Agentは変更しない。

公式ルール確認: https://service.mattel.com/instruction_sheets/FWG43-Eng.pdf (全6ページ、開始点図 p3)。14×14、21種、同色角接触/辺禁止、相手接触自由、強制pass、残マス負点、全置き15/mono最後20。座標は依頼どおり(4,4)/(9,9)。
PyTorch公式 https://pytorch.org/blog/pytorch-2-7/ はBlackwellとCUDA12.8 wheel対応を明記。Python3.14に対応するcu128版を専用venvに導入し実CUDAで検証予定。
実機 E-frontier / RTX5090 32607MiB / driver595.91.07 / Python3.14.4。

設計: free polyominoを1〜5マスまで列挙・D4同値で正規化。全配置をbitboardで事前生成し、角候補から合法手列挙。独立checkerは座標集合で判定。候補特徴のMLP policy + state value。heuristic教師で初期化後、終局報酬actor-critic自己対戦を実装。候補を全件評価し枝刈りなし。seed/optimizer/RNGを保存しゲーム境界で再開。先後交替評価とCPU生成/GPU時間を記録。

環境作成: system pythonのensurepip欠落を検出。専用venvの代替作成方法を確認中。

エンジン/AI/CLI/テスト実装。候補特徴12次元＋残ピース等状態46次元、MLP policy/value。教師あり初期化は明示的にimitationフェーズ、後段は終局得点差/109のMonte Carlo actor-critic自己対戦。手工芸heuristicは大駒優先＋新規角＋相手角遮断＋進出。独立checkerで全盤配置総当たりとの集合一致も検証する。
uvが選択した専用venv Pythonは3.12.14（systemの3.14とは別）。インストールログはinstall.log。

秘書から独立検証報告: 別実装座標set oracle、40対局1208手・56局面の全合法手集合一致、保存則/終局/得点PASS、初手414通り。レビューのコードは村側にあり当方は変更しない。
依頼を反映: imitation.ptを模倣終了時に別保存、trainデフォルト200局へ変更、未学習/模倣/自己対戦を個別評価する。
最新自動選択torch2.11のNVIDIA依存取得がtimeoutしたため、公式Blackwell対応系列のtorch2.7.1+cu128に固定しインストール成功。requirements.lock.txtに全依存を保存。

実行結果: `.venv/bin/python -m pytest -q` → 7 passed in 4.40s。
`.venv/bin/python -m blokus gpu-check` → RTX5090 / sm_120 / torch2.7.1+cu128 / CUDA12.8、実forward/backward、全parameter勾配finite、peak allocation 73,095,168 bytes。詳細 artifacts/gpu.json。
実学習開始: `.venv/bin/python -m blokus train --games 200 --seconds 180 --imitation-games 60 --seed 42 --output artifacts/run`。初回予算を180秒に制限。完了後に模倣と自己対戦を区別して評価。
評価seed1000〜1009、相手ごと20局、先後10局ずつ。heuristicと未学習モデルを先に評価中。

初回200局=13.315秒。選択用seed1000評価（各相手20局）:
- heuristic対random 20勝0敗0分/+27.30、対heuristic 6勝6敗8分/0。
- 未学習: random20勝0敗/+47.70、heuristic10勝10敗/+2.35。
- imitation60: random20勝0敗/+28.40、heuristic4勝16敗/-5.20。
- selfplay200: random20勝0敗/+25.90、heuristic10勝10敗/-2.20。
改善の証拠なし。初期random weightsが偶然有用な特徴重みを持つため未学習も強い。この比較を省略しない。

秘書の追加依頼で予算内の改善試行: 教師の全候補softmax分布によるcross entropy＋中心化logit回帰0.1に変更。単一argmax分類より候補間の順位情報を利用。模倣2000局、自己対戦2000局、自己対戦lrを1e-4へ低減（模倣1e-3）。4000局または360秒の早い方で停止。累計学習は600秒未満に制限。選択seed1000は開発用、最終評価は未使用seed2000以降で行う。

秘書追加検証: 別processのCUDA連続4局 vs 2+保存再開2局でモデル/Python/Torch/CUDA RNGがbit一致。結果 artifacts/secretary-review/resume/result.json。エンジンsha256も独立検証時と一致。
実用修正: evaluate既定出力をartifacts/evaluation.jsonに分離、親dirを作成。trainはconfig.jsonとconfig-history.jsonlに全引数/seed/torch/CUDA/Python/GPU/source SHA256を保存（後者は追記）。この追加前に起動済みの初回と追加学習のconfigは事後記録として明示する。

追加模倣2000局の選択用評価: 対random20勝0敗0分/+25.50、対heuristic2勝10敗8分/-1.20。初回模倣よりheuristicへの得点差は縮小したが未学習には届かない。自己対戦継続中。
最終比較はseed2000〜2049、相手ごと100局（先後50ずつ）。モデル選択はseed1000の対heuristic勝率、次に平均得点差を優先し、最良学習済みと最良全体を区別する。最終seedの結果ではモデルを再選択しない。
CLI検証: AI対AI（heuristic vs random、seed7）完走 -19/-37。人間入力経路へindex0を反復投入し終局まで完走。GPU再開2局も成功。pytestは10 passed。

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

初版完了。最終pytest: 10 passed in 2.75s。推奨学習AI対heuristic CLI完走（-15/-12）。専用ディレクトリのみ変更。長時間学習プロセスなし。README、全checkpoint、JSON評価、環境lock、独立検証資料を保存。

# 継続研究 round1: three-methods

開始: 2026-11-05。基点 main 7db897e343089c8576fb04438cd91431778fb024。branch research/three-methods。新規Agentは使わず専用repoのみ変更。engine.pyは変更禁止としてhashを継続検証。GPU初期負荷0%、206MiB、他学習なし。

## 設計・API（実装前記録）
- 新規 `blokus/research.py`: `encode_board(State)`（手番基準の自石/相手石/自開始点/相手開始点の4x14x14）、`action_mask(State)`（全13729配置+pass）、A/CNNモデル、`Agent.act(state,rng)`、状態clone、終局・手番基準value、beam negamax探索。盤面と残駒46特徴を入力し、CNN候補は配置セルの空間特徴を集約。合法候補だけ採点、passは合法手なしのみ。
- 新規 `blokus/experiment.py`: `python -m blokus.experiment {run,train,tournament,play,prefixes}`。学習/評価/CLIはv1から分離。
- A: 軽量候補MLP、教師の順位分布を安定に蒸留し、heuristic/random/v1/自分の相手pool上の対局報酬による小さなactor-critic補助を併用。heuristic priorを明示し、未学習Aも同じpriorを使う。
- B: 盤面CNN＋残駒＋候補特徴policy/value。Aと同じ教師/相手pool。初期モデルと学習後を比較。
- C: Bと同構造だが独立seed/checkpoint。幅制限2ply negamaxの探索教師を蒸留。探索shortlistに更新student policyも使用するexpert iteration。C-search-offを主な学習済み比較、C-search-onを思考時間追加の別条件。探索は固定node予算であり時間同等とは主張しない。
- 共通学習上限: 各180秒、ゲーム境界で最大1ゲーム超過。教師生成/対局/更新/保存を含む。同一GPU・逐次学習。各中間/最終checkpointをvalidationで選択。追加改良はvalidationだけで判断し別予算として記録。
- prefix: 学習と別seedで合法ランダム2〜6plyを作成しJSON保存。validation seed41000、holdout seed51000、初回各12/24prefixを予定。全対戦で同じ集合、各prefix2局でAgentの色を交換。色盤面は固定し役割だけ交換。集合と各prefixにSHA256、重複なし/分割重複なしを検証。学習prefix seedは別系列。
- selection: validationでheuristic/v1へのpaired平均得点差平均を最大化。各方式中間/最終から固定しselection.jsonに記録後、holdoutを一度だけ評価。holdout結果で再選択禁止。
- holdout: A/B/C-offとheuristic/v1/random、A/B/C総当たり、各未学習vs学習後、B vs C-off、C-off vs C-on等。prefix単位のpaired平均をbootstrapし95%区間（得点差と勝+0.5分率）。2局を独立サンプルと数えない。24prefixは暫定。
- config/seed/source hash/torch/CUDA/モデルmetadata/optimizer/RNG/対局数/更新数/学習秒/推論ms/探索nodesを保存。baseline v1ファイルとheuristic/random設定を凍結コピー。
- テスト: board tensor、合法action mask/pass、相手手番value符号、終局utility、探索入力不変性/合法性、prefix再現/paired集計、保存再開。

秘書向け: engine APIは変更しない。新規APIは上記2ファイル。成果物は `artifacts/research-round1/`。実装中に細部の変更・実測・選択基準の適用を追記する。

設計補足: 秘書提供のfrontier高速化 e5717e1 をcherry-pick（当branch a4d5a0d）。ルール/API/列挙順序は不変、独立検証PASS報告あり。以後この版を共通使用し、エンジンのそれ以外は変更しない。
研究stateを49次元へ拡張: 既存46特徴+手番側score/89+相手score/89+passes/2。全置き15/20点の差とpass履歴を保持。v1入力は46のまま互換維持。CNN全盤面flatten＋候補セルconv特徴を併用。
参考確認: https://arxiv.org/abs/1705.08439 （探索とpolicy一般化を交互に改善する発想）、https://arxiv.org/abs/2108.13264 （点推定のみでなく区間を報告）。本実装は幅制限2plyの簡略探索で、論文のMCTS再現ではない。

実装完了: research.py（モデル/search/agent/checkpoint）、research_train.py（共通予算学習）、tournament.py（prefix/paired bootstrap）、experiment.py（CLI/run orchestrator）。22テストPASS（3.17秒）。B/C実CUDA2局smoke成功。Cは352探索nodeを実際に生成し更新。
本実験開始コマンド: `.venv/bin/python -m blokus.experiment run --seconds 180 --validation-prefixes 12 --holdout-prefixes 24 --output artifacts/research-round1`。各方式180秒、逐次。A教師はheuristic logits + 2×標準化v1 logits、Bも同じ。Cは探索評価×40を探索候補内softmax。A/Bのactor-critic係数0.15、value回帰0.5、entropy0.002。全方式heuristic prior+ゼロ初期残差を明示。

A完了: 4104局/4104更新、180.046秒、57,302学習局面。validation平均（heuristic/v1）: 中間-1.125、最終+0.3125で最終を選択。これはholdout結果ではない。B学習へ移行。
保存再開テストをA/B/Cすべてに拡張し24テストPASS（4.15秒）。独立監査 scripts/audit_research.py は秘書作成のまま保持し最終commitへ含める。

負荷観測: B学習中CPU約100%（1コア）、GPU17%、GPUメモリ965MiB、約98W。GPUのみを飽和させる実装ではなく、逐次対局生成と小さな更新のCPU/GPU往復が多い。environment-gpu.txt / environment-cpu.txtを保存。

秘書から次ラウンド提案: 現A/Bは同一教師蒸留が強く、教師を超える信号が弱い可能性。現runは変更せず完走。次の候補A2は候補特徴重みをpaired対局成績で直接最適化するCEM/進化戦略（CPU対戦ベース学習でGPU勾配学習とは区別）。実施時は学習prefix/validation/新holdoutを分離し、初回holdout後の設計変更に旧holdoutを再利用しない。

B完了: 3325局/更新、180.028秒、46,597学習局面。validation criterionは中間+0.1042、最終+1.3333で最終を選択。最終対heuristic平均+2.6667（prefix bootstrap95%区間[0,5.25]）、対v1平均0（区間[-2.125,2.251]）。現時点では明確な強化成功とは言えない。C探索教師学習へ移行済み。

C完了・3方式選択固定。C validation criterion: 中間-1.3333、最終-0.3333、最終を選択。selection.jsonを書き出した後holdout評価を開始。これ以降のモデル再選択はしない。
学習時の全ソースをrun-config hashと一致確認して `artifacts/research-round1/source/blokus/` に凍結保存。学習後の運用修正は(1)CLI出力defaultをsubcommandごとに分離、(2)resume時rollout/update秒の累積表示を保持、の2点のみ。モデル/学習式/評価動作は変えず、実行中runは読み込み済みの凍結ソース版のまま継続。再現時に厳密なソースが必要ならsource snapshotを使用。

Round1評価完了: train A4104/B3325/C2205局、各180秒、C教師探索457,156node。run全体582.76秒。selection固定後、24prefix×2局×21組=1,008holdout局。
探索なしA/B/Cのbaselineへの得点差CIは0を含み、明確な勝者不明。総当たり観測point率はA>B>Cだが暫定。C-search対heuristic32/14/2、平均+3.00 [0.77,5.46]、対v1 31/16/1、+1.875 [0.083,3.708]。C-offとの差は+1.19 [-1.08,3.19]でまだ不明。C-search3.262ms/手 vs C-off0.963ms/手。思考時間同等とは主張しない。
独立監査実行: `.venv/bin/python scripts/audit_research.py artifacts/research-round1/holdout-results.json artifacts/research-round1/holdout-prefixes.json` → PASS。report.mdに全表・区間・計測内訳を記録。全重み保持、holdout再選択なし。

# 追加改善 A2 CEM（村長/秘書承認）
第一ラウンドの21match完了・保存後に開始。秘書独立監査: 24unique開始状態/1008対局、selection、全A/B/Cの別processCUDA再開bit一致PASS。資料 artifacts/secretary-review/round1-audit.json と research-cuda-resume/result.json。
A2計画: CPUで候補特徴の線形重みを対局成績から直接最適化。GPU勾配学習ではない。population10、各世代train prefix3個、heuristic/v1相手、各prefix先後交換。世代seed71000+generationで更新し、全候補で共有。10候補×3prefix×2色×2相手=120対局/世代。CEM上位3候補から平均/標準偏差を平滑更新、incumbentと中心候補を含める。
追加学習目安240秒、世代を途中で切らない。最大100世代、次世代の見積時間（直前×1.25）が残予算を超える場合は開始しない。初回世代は必ず完走し実測、世代が240秒を超える場合はその1世代で停止。世代ごとの実時間・対局数・候補数・train prefix hash・重み・fitnessを記録。
checkpointは初期/各10世代/最終を保存。既存validation41000の12prefix・heuristic/v1への平均得点差で選択。新holdout61000の24prefixを事前固定し、選択後はA2/heuristic/v1/B（元validation最良）/C-searchを総当たり。旧holdout51000は使用しない。時間不均等・CPU学習である点を明記。

A2実装・開始: blokus/cem.py、Agentのformat3 JSON線形モデル対応、tests/test_cem.py。追加2テストPASS（2.27秒）、CEM連続2世代と1+保存再開1世代の重み/平均/標準偏差/RNG/対局数が一致。
実測1世代約2.3秒、GPU0%/177MiB。学習はCPUのみ。係数の正スケール不定性を避けサイズ係数12固定、候補内で定数な2特徴を除いた9係数を最適化。実行: `.venv/bin/python -m blokus.cem run --seconds 240 --generations 100 --population 10 --train-prefixes 3 --output artifacts/research-a2`。

A2新holdout開始前の追加比較宣言: A2-searchをA2-off/C-search/heuristic/v1と比較。runは起動済みのため別 `search-results.json` に4組保存する。`search-ablation-plan.json` 作成時holdout-results.jsonが未存在であることをassertし記録。同一A2-off validation選択checkpoint、幅4×3、depth2、holdout再調整なし。再現スクリプト scripts/evaluate_cem_search.py。

最終追加テスト: A2 JSON+searchで合法手・入力State/重みファイル不変・node上限16を確認。全27テストPASS（5.70秒）。A2実験sourceもrun-config hash一致を確認してsource/blokus/へ保存。A2-search追加は新holdout閲覧前に宣言済み、同一重み固定。

A2追加実験完了: CPU237.886秒、96世代、11,520対局、960候補評価。validation criterion+2.3125の90世代目を選択。selected.json SHA d7b5049aaa96ebbd2d73d8ba45d727cbac6ee5ed0dbba857c33fa078044fccdd。新holdout SHA a0f2237c966505bcb11ff9e753885ca08f49101f9b52a8068071ff4916964e06。
新holdout通常11組+追加探索4組、計720局。A2 vs B 29/16/3、平均+2.9167 [1.2083,4.7292]。A2-search vs C-search 28/19/1、+1.7083 [-0.4583,4.0417]で優劣不明。A2-off0.545ms/手、A2-search2.967ms/手。両レポートを保存。
監査: 学習prefix288個と評価prefix集合の重複0、全960候補のfitnessを各対局得点から再計算一致、validation最大値選択/重み一致、両source snapshotのhash一致。selection-training-audit.json保存。通常/探索両holdoutを秘書作成audit_research.pyで監査PASS。
全27テストPASS、A2-searchの人間入力CLI終局(-59/-18)、実行プロセスなし・GPU0%。秘書のseed81000 heuristic-search追加対照は診断専用で選択に使わない。追加開発を広げず、この診断を受けてcommit/pushする。

秘書探索対照診断を受領（seed81000、24unique prefix、192局、audit PASS）。C-search vs heuristic-search:27/19/2、+1.50 [-0.67,3.65]。A2-search vs heuristic-search:16/28/4、-0.54 [-2.75,1.58]。A2-search vs C-search:24/22/2、+1.27 [-0.65,3.10]。heuristic-search vs heuristic:30/15/3。学習なし探索も強く、探索あり方式の優劣・学習寄与は断定不能。診断でcheckpoint再選択はしない。
成果物すべて保存、最終27テストPASS、CUDA各方式finite勾配/全optimizer・RNG再開一致、独立監査資料を含めてbranchへcommit/pushする。mainは変更しない。

# 再開: 終局価値学習と同一思考時間探索
基点 main 69065fc、branch research/value-search。既存ルール・旧checkpointを保持。他Agent/村本体は変更しない。GPU初期負荷0%、177MiB。

計画: (1)現行CPU手生成・特徴・静的探索/GPU推論を短く計測。(2)実ゲーム終局得点差を教師にするbatched value modelを実装。静的評価の残差を学習し、盤面・残駒・到達可能角等から将来の置き残しを予測。(3)同一wall-time予算のanytime探索を共通実装し、非学習static、旧C-policy+static、学習valueを比較。(4)最低3学習seed。validationで価値混合率/モデルを選び、選択固定後fresh holdout。prefix paired/bootstrap、実思考時間・超過・nodesを保存。(5)最初の結果が弱ければ分布ずれ/予測精度/時間配分を診断し、独立した修正実験を行う。
GPU学習合計上限1800秒。まず小規模3seed、学習をバッチ化しCPU生成とGPU更新を計測。巨大な中間checkpointは保存せず各seed latest/selectedのみ。実装・プロファイル後に具体的予算と分割seedを固定する。

実装: value_search.py（6plane盤面/49state、色交換反対称の終局価値残差CNN、共通6ms wall allowance探索）、value_data.py（spawn CPU6worker、実終局ラベル、枝局面は独立に終局まで継続）、value_train.py（batched CUDA回帰・game単位診断split・完全resume）、value_experiment.py（2段階run）。5追加テストPASS: antisymmetry、全置きbonus/pass/終局、探索不変性/合法性、遅延child棄却、optimizer/RNG再開一致。
初期profile: heuristic0.620ms/手、固定幅static-search2.849ms/手、C-search3.717ms/手（初回GPUwarm-up影響あり）。6ms共通予算は生成/特徴/順位/GPU同期を含む。次child所要時間を推定して開始を控え、deadline後のchild結果は使用しない。ただしnative処理自体を強制中断できないため実超過を記録し、hard realtime保証とは呼ばない。
分割: train110000〜、追加branch120000〜、validation91000(12prefix)、初回holdout92000、修正後fresh holdout93000(各24prefix)。seed201/202/203、stage1各1600更新/120秒上限、修正stage2各2400更新/120秒上限、GPU学習上限合計720秒＋smoke。選択はvalidation上のstatic/C両対照への平均得点差、alpha=.25/.5/1。stage2はbranch labels追加＋remaining合計24以下でvalueを使うgate。全seed両対照の得点差CI下限>0でなければ初回で止めず修正へ進む。

価値学習stage1 seed201: 1600更新/2.17秒、validation criterion -2.125、alpha=0.5、gate=42。

価値学習stage1 seed202: 1600更新/1.95秒、validation criterion -0.688、alpha=1.0、gate=42。

価値学習stage1 seed203: 1600更新/2.01秒、validation criterion +1.625、alpha=0.5、gate=42。

Stage1で全seed・両対照へのCI正を達成せず。診断JSONを保存し、探索leafを独立に終局まで継続した追加データと後半局面gateでstage2へ進む。旧holdout92000は再選択に使わず、新holdout93000を使用。

価値学習stage2 seed201: 2400更新/2.95秒、validation criterion +0.354、alpha=1.0、gate=24。

価値学習stage2 seed202: 2400更新/2.95秒、validation criterion -0.583、alpha=0.25、gate=24。

価値学習stage2 seed203: 2400更新/2.95秒、validation criterion -0.562、alpha=0.25、gate=24。

Stage1/2実験完了（run wall287.63秒、GPU学習合計14.99秒）。初回1,800対局55,238局面、収集23.38秒。flat CNNの学習lossは低下する一方、holdout-game MSEが悪化（static約.0394、各seed初期best約.038〜.042、最終約.049〜.055）。過学習と推論のrootごとの往復を疑う。Stage2のbranch追加＋late gateでも安定改善なし。旧ソースをrun/sourceにhash一致で保存し、ここで止めずstage3へ。
Stage3計画: pooled CNN（全盤flattenを2x2 poolingへ縮小）、開始点集合を保つtranspose/180度回転の対称性拡張、CPU追加6,000終局対局、同じ3seedをscratch学習。複数rootの葉をまとめて1バッチ評価する共通探索schedulerをstatic/C/valueすべてに適用。追加GPU上限各120秒×3（累積上限1080秒＋smoke、30分以内）。validation91000で固定後、fresh holdout94000。旧92000/93000では選択しない。

Compact stage3 seed201: 4000更新、5.17秒、best predictive MSE=0.03512、validation criterion=+1.083。

第三修正: AdaptiveAvgPool2dのCUDA backwardはstrict deterministic非対応だったため、14→2の同等な固定AvgPool2d(7,7)へ修正。生成済み6000局のhashを検証して再使用し、失敗時run-configを保全。決定性を緩めず再学習中。

Compact stage3 seed202: 4000更新、5.27秒、best predictive MSE=0.03493、validation criterion=+1.938。

Compact stage3 seed203: 4000更新、5.24秒、best predictive MSE=0.03389、validation criterion=+2.083。

第3段階の監査: pooled予測MSEは改善したがbatched schedulerのoverrun最大約20%、fallback多発を発見。推奨から除外し、固定重みを維持してincremental schedulerへ戻す第4段階を開始。validationのみでalpha選択、fresh holdout95000、全3seed。GPU再学習なし。全34テストPASS、flat/pooled各CUDA別process 4更新 vs 2+2再開でmodel/optimizer/Torch/CUDA RNG bit一致（augmentationあり）。

原因分析（検証済み事実と仮説を区別）: flatはtrain loss低下に対し未使用対局MSE悪化、過学習を観測。pooled+追加データではMSE改善を観測したが対戦改善なし。終局回帰は継続方策の期待得点を学ぶため、候補間の微小な順位差・相手の最善応手の学習とは一致しない。batched版の時間超過/破棄も実測された。次案は同一局面の複数候補を同一継続乱数で完走し、差分/ランキング損失と局面単位検証を導入すること。現在の反対称制約は手番の先行利益も抑えてしまう可能性があり、手番の扱いを分離したablationが必要。これらはまだ実証していない仮説。

最終第4段階完了: fresh95000/24prefix/3seed、非学習探索に平均+2.146 CI[+0.291,+3.875]、時間制御版Cに+1.194 CI[-0.500,+2.903]。学習側4.575〜4.690ms/手、6ms超過0〜0.80%。安定した優越は未確定。validation最高criterionのseed201/alpha0.5/gate24/incrementalを試用例としholdout再選択なし。全34テストとCUDA別process再開、全28match算術監査、train/eval開始状態非重複、CLI終局を確認。GPU主学習計30.672秒/24000更新、全run有限終了。詳細 artifacts/value-search/conclusions.md。research/value-searchへcommit/push準備。

## 正式子依頼: 同一局面候補の差分/ランキング学習
基点1b5cf87、branch research/paired-ranking。requests getでwork/runのacceptedを確認。完了時は最新revisionを取得しrequests finishを登録する。
設計: 新規seedの対局から終盤寄りの共通rootを抽出。各rootで4候補を2ply後の葉に変換し、各候補を同じ継続seed・同じ方策で複数回終局まで進め、最終得点差を取得。候補ごとの独立なラベルを使い、group内差分回帰＋soft ranking＋弱い絶対値回帰でpooled価値CNNを学習。共通seedは初期乱数を共有するが、合法手や乱数消費の分岐後まで同一とは主張しない。
3seedは同じ既存value checkpointから初期化し更新乱数を変える。既存モデル/エンジンを保持。新validation seed210000/32prefixで混合率とcheckpointを固定してからfresh holdout220000/96prefixを生成。先後交換、非学習探索と既存value(seed201,alpha0.5,gate24)へ共通6ms incremental探索で比較。prefix内2局をまとめたCIとseed×prefix CI、全decision時間、超過/node数を記録。
初回予定1200base対局×2root×4候補×2完走=19200候補継続、CPU6worker。GPU各seed最大180秒・合計540秒以内（上限30分）。失敗も保全して原因を記す。全体2時間以内に中間成果を確定し、無期限runは残さない。

Ranking seed301: 9.27秒/6000更新、validation criterion=+1.977, alpha=0.5。

Ranking seed302: 8.93秒/6000更新、validation criterion=+1.492, alpha=1.0。

Ranking seed303: 8.95秒/6000更新、validation criterion=+1.867, alpha=1.0。

時間制御の不具合を検出: seed303/priorの探索時間1.623ms/手。一度の大きな遅延でcost_estimateが予算より大きくなり、以後のchild試行が全拒否され推定値も更新されない。評価劣化を学習原因とは判断しない。旧結果/sourceを保全し、観測費用と推定値を予算基準で上限化。100ms擬似遅延の次手に探索が回復する回帰テストPASS。全モデル・alphaは固定し、全対照へ同じ修正を適用。再選択せず新holdout230000/96prefixで全7組再比較する。

正式子依頼の最終結果: repaired共通6ms、fresh230000/96prefix、3seed平均で非学習探索+2.583 CI[1.773,3.325]、現value -0.066 CI[-0.688,0.578]。既存valueからの改善は示せず推奨を維持。主学習27.16秒/18000更新、CPU19200継続/52.38秒、初回+修正版実験run617.86秒、3840評価対局。全38テスト、CUDA別process4vs2+2 model/optimizer/全RNG bit一致、256終局ラベル再生、選択/時間/hash/開始状態非重複の監査PASS。CLI終局PASS。artifacts/paired-ranking/conclusions.mdに原因と次案。research/paired-rankingへcommit/pushし、正式requests finishで秘書へ引き継ぐ。

main統合レビュー: origin/main=69065fcに対する1b5cf87/3dc3eb6の差分を確認。既存engine/既存学習APIへの変更なし、新規value/ranking経路とテスト追加。終局価値符号、group分割、RNG再開、時間費用上限、旧結果保全を確認。全38テスト6.17秒PASS。origin/mainをfetchし他者更新なし、3dc3eb6までfast-forward。README冒頭を既存value推奨/paired-ranking改善未確認へ更新して通常pushする。

## 実探索分布に合わせる新実験（正式子work46f31ded）
branch research/search-aligned、基点3dc3eb6＋main README整備。上位4root候補×相手heuristic上位3応手を全て保持し、masked葉をgroup化。継続poolは非学習固定4×3探索と既存valueの固定4×3探索、双方同方策と交差対戦を4条件で循環。時間予算によるラベル非決定性を避け、継続生成はCPU固定幅とする（評価の6ms探索とは区別）。各plyで乱数を再seedし、候補/応手間で同じ継続条件と同じ相対ply乱数を共有。合法手が違うため同じ乱数が同じ行動を意味するわけではない。
先に24group×12継続のpilotで2/4/8完走平均を別の4完走基準と比較し、候補順位/意思決定損失/ラベル差を確認。trainは新seed300000から共通groupデータ、repeatsをpilotで選定。3seed501/502/503、共通事前学習・データ・minibatch/augmentation乱数、損失だけabsolute/difference/rankingで比較。各6000更新/180秒上限、9学習最大1620秒でGPU30分内。新validation310000/32prefixで方式を3seed平均対戦成績から固定、alpha0.5/gate24共通。選択方式の3seedのみfresh holdout320000/96prefixで現valueと非学習探索に先後比較。旧holdout不使用。全体2時間を上限目安に記録する。

Aligned absolute/501: 6000更新/10.88秒、best decision regret=0.686点、validation直近2相手mean=+1.211。

Aligned absolute/502: 6000更新/10.72秒、best decision regret=0.587点、validation直近2相手mean=+0.922。

Aligned absolute/503: 6000更新/10.62秒、best decision regret=0.634点、validation直近2相手mean=+1.633。

Aligned difference/501: 6000更新/12.24秒、best decision regret=0.654点、validation直近2相手mean=+1.047。

Aligned difference/502: 6000更新/12.27秒、best decision regret=0.647点、validation直近2相手mean=+1.297。

Aligned difference/503: 6000更新/12.24秒、best decision regret=0.663点、validation直近2相手mean=+0.883。

Aligned ranking/501: 6000更新/12.38秒、best decision regret=0.659点、validation直近2相手mean=+1.234。

Aligned ranking/502: 6000更新/12.31秒、best decision regret=0.653点、validation直近2相手mean=+1.383。

Aligned ranking/503: 6000更新/12.22秒、best decision regret=0.686点、validation直近2相手mean=+1.211。

pilot完了: 24group/12継続、独立な追加4継続に対する意思決定損失は2/4/8完走で0.833/0.469/0.188点、候補MSEも減少。事前ルールで8完走を採用。64ラベル再生PASS、上位3応手/mask/保存再開の新規3テストPASS。main remote eb96a90を確認。
全9モデルのvalidation完了: absolute +1.2552、difference +1.0755、ranking +1.2760（3seed×2相手平均得点差）。rankingを固定。ただしabsoluteとの差0.0208点は小さく、方式の真の優越とは主張しない。選択後にfresh320000/96prefixを生成し、ranking3seedのみ最終評価中。

search-aligned完了: fresh320000/96prefixで選択ranking3seedは非学習+3.113 CI[2.339,3.922]、既存value+0.276 CI[-0.314,0.884]。追加改善未確認、既存value推奨維持。実戦混合後診断はprior regret0.747点/順位76.78%、ranking平均0.779点/74.41%で改善なし。主学習9モデル54000更新/105.88秒、CPU103448継続/258.18秒、run全体725.06秒＋pilot15.26秒。2496評価局。全41テスト、3損失CUDA別process4vs2+2の重み/optimizer/全RNG bit一致、pilot64＋full64ラベル再生と選択/hash/時間/新旧prefix分離監査PASS。CLI終局PASS。main eb96a90通常push済み。新実験はresearch/search-alignedへcommit/pushしwork46f31dedのrequests finishで秘書へ返却する。

## 訪問局面の多様性と実戦checkpoint選択（work730484b4）
基点9e76141、branch research/visited-diversity。mainは変更しない。requests getで今回run activeを確認。
事前比較条件: 損失はabsoluteのみ固定。各更新時点の同じ重みからraw基準と実戦基準(alpha0.5＋終局厳密上書き)の両checkpointを保存し、新validation410000で比較。3更新seed601/602/603、同一初期value、共通の独立診断root256/8継続。実測6ms探索の完了した子評価を捕捉し、実際のrootと最大4候補×3応手の葉を保存。時間打切りがあるので訪問集合のbit再現は保証せず、保存後のラベルは決定的な固定幅pool継続で再現する。
pilot多様性条件: wide512root×4継続 vs dense128root×16継続、denseは残駒数で層化したwideの部分集合。4条件poolを完全に周回し方策比率を一致。公称葉完走数を同程度にし実継続数/ply/CPU時間も記録。pilot両条件×3seed、3000更新/90秒各。共通診断で実戦損失が初期値より平均10%以上低下し、validationで既存valueへの平均差が+0.25点以上の条件がある場合のみ拡大。これは拡大判断用の小規模gateで優越の証明ではない。信号なしなら全量を使わずpilot結果を確定する。
拡大時は最大10000wide/2500dense root、4/16継続。pilot実速度から両条件のラベル生成合計35分以内に収まるroot数へ下方調整可。追加各6000更新/120秒、GPU主学習上限pilot540＋full720=1260秒。方式・selectorをvalidationで固定してからfresh holdout450000/96prefixで選択条件3seedのみ対既存value/非学習を先後交換・6ms比較。旧holdout不使用。
別診断: 固定既存valueの6/25/100msを開発prefix400000/8組で役割交換の直接対戦。学習と混同せず非等時間診断として記録。root/評価葉/元対局の全中終盤state hashと評価対局のstate hashを監査。ラベル継続の全内部状態までは保存しない限界を明記。全体2時間を監視する。

Visited pilot/wide/601: 3000更新/4.92秒、dev best raw/live={'raw': (0.9028320908546448, 0.006463458761572838), 'live': (0.9150390625, 0.006223971955478191)}。

Visited pilot/wide/602: 3000更新/4.69秒、dev best raw/live={'raw': (0.8725586533546448, 0.006426596082746983), 'live': (0.8867186903953552, 0.0063035800121724606)}。

Visited pilot/wide/603: 3000更新/4.86秒、dev best raw/live={'raw': (0.91357421875, 0.006367459427565336), 'live': (0.96142578125, 0.0062548937276005745)}。

Visited pilot/dense/601: 3000更新/4.96秒、dev best raw/live={'raw': (0.8735352158546448, 0.0066450415179133415), 'live': (0.83056640625, 0.006326326169073582)}。

Visited pilot/dense/602: 3000更新/4.96秒、dev best raw/live={'raw': (0.8266600966453552, 0.006585485301911831), 'live': (0.76904296875, 0.0063097476959228516)}。

Visited pilot/dense/603: 3000更新/4.96秒、dev best raw/live={'raw': (0.82470703125, 0.006628886796534061), 'live': (0.8359375596046448, 0.006323458161205053)}。

### Visited diversity pilot判定
新devの初期live regret=1.0298点。wideの3seed best平均0.9211（10.56%減）、dense=0.8118（21.16%減）。新validationでlive対既存value平均はwide +0.5625、dense +0.4479点。事前gateを両条件が通過したためfullへ進行。生成見積り35分からwide9256 root×4、dense2314 root×16を固定（目安1万から計算予算内へ調整）。全量のvalidation seed411000と最終holdout450000はまだ選択に未使用。pilotだけで改善とは結論しない。

### Full生成中
実6msゲームから9,256unique rootの収集を完了。wide4完走を6CPU workerで生成中（約8,100rootで846秒、計画範囲内）。その後同じ収集集合の層化subset2,314rootを16完走する。GPU主学習の追加前であり、full validation/holdoutは未参照。検証用にscripts/verify_visited.pyを用意し、計測終了後に全テスト・別process CUDA再開・全評価着手再生・ラベル再生・CLI終局を一括実行する。source/root/leaf状態hashの監査は完全重複に限り、類似局面や全ラベル継続内部の非重複を保証しない。

### Fullデータ完成・学習開始
wide9256×4: 396,532継続、3,731,669ply、967.23秒。dense2314×16: 395,952継続、3,750,128ply、976.48秒。CPU6workerで生成時間差約0.96%、ply差約0.49%に収まり、概ね同計算量の比較になった。各条件3seed・6000更新・同一absolute損失で学習中。new validation411000/32prefixでraw/liveと条件を選択し、その後fresh450000/96prefixへ進む。

Visited full/wide/601: 6000更新/9.95秒、dev best raw/live={'raw': (0.8076171875, 0.00652588065713644), 'live': (0.81787109375, 0.006239873822778463)}。

Visited full/wide/602: 6000更新/9.89秒、dev best raw/live={'raw': (0.91845703125, 0.006492690183222294), 'live': (0.84716796875, 0.006262396462261677)}。

Visited full/wide/603: 6000更新/9.53秒、dev best raw/live={'raw': (0.8305663466453552, 0.006538674235343933), 'live': (0.8427734375, 0.006292452570050955)}。

Visited full/dense/601: 6000更新/9.84秒、dev best raw/live={'raw': (0.88427734375, 0.0066073015332221985), 'live': (0.9052734375, 0.006380051374435425)}。

Visited full/dense/602: 6000更新/9.44秒、dev best raw/live={'raw': (0.8359375, 0.006659537088125944), 'live': (0.88330078125, 0.006325321272015572)}。

Visited full/dense/603: 6000更新/9.45秒、dev best raw/live={'raw': (0.892578125, 0.006645632442086935), 'live': (0.90283203125, 0.006340876221656799)}。

### 最終方式固定
full validation411000/32prefixで、dense/raw（3seed×static/priorの平均criterion +2.02604点）を選択。全条件のvalidation結果とselection.jsonを保存し、final-selection.jsonを書いた後にのみfresh450000/96prefix評価を開始した。方式・checkpoint・alpha=.5・gate24・6ms予算は以後固定。live選択やwide条件を新holdoutの結果で再選択しない。

### 完了・結論
fresh96prefix/先後交換/共通6msの1344局を完走。3seed交差bootstrapで対非学習+2.354 [1.542,3.196]点、対既存value −0.099 [−0.781,0.477]点。追加改善は未確認、既存value推奨を維持。live選択はfull dev regretを低下させたが、新validationでraw選択を上回らなかった。wideの元ゲーム数815、dense780で、root4倍は独立展開4倍ではない。次は収集ゲーム単位の相手/展開多様化と候補外・終盤の教師精度を調べる。探索時間曲線は8prefixで全得点CIが0を含み、拡大確認後に深さ・終盤完全探索の予算配分を検討する。

GPU主学習54,000更新/87.47秒、準備＋run3437.48秒。CPUラベル生成（dev除く）837,440継続/2059.84秒。全44テスト6.70秒、別process CUDA model/optimizer/Python/NumPy/Torch/CUDA RNG bit一致、CLI終局PASS。監査591.99秒で全評価着手・得点・CI・時間・checkpoint選択・ラベル標本再生PASS。109,619学習root/leaf状態、25,623収集中終盤状態をdev/evaluationとhash照合し完全重複なし（全継続内部と類似局面は未監査）。再実行時の監査上限は余裕を持って1200秒へ設定、初回の591.99秒監査は当初600秒上限内で完走した。

README、artifacts/visited-diversity/{METHOD.md,conclusions.md,compute-summary.json,audit.json,verification.json,diagnostics.json}に結果・限界・コマンドを保存。研究branchのみcommit/pushし、mainの既存value推奨は変更しない。学習/評価の残存runなし。

保存時に1MB超のJSONのみ空白を除去し、再decode値の完全一致を確認。元/保存後hashとサイズをartifact-storage.jsonに記録した。checkpoint/NPZのbytesは変更していない。成果全体は約90MB、checkpoint合計約4.2MB。監査後の差分はレポート生成・文書・この意味保存のJSON整形のみで、engine/学習/探索コードは検証時から不変。

## 固定重み探索強化（基点a88905e）設計
追加学習なし。既存value（compact/seed-201/predictive-best.pt、alpha=.5、gate24）とengineを固定。新規blokus/deep_search.pyへnegamax alpha-beta、深さ2/4/6…の反復深化、heuristicによる固定候補幅、TTのEXACT/LOWER/UPPERを実装する。候補を絞った探索は選択探索と呼び、厳密探索とは区別する。少残駒時は全合法手・終局までの探索を試み、完了した場合だけ厳密解と記録する。期限内に完了した反復だけを採用し、中断中の値は使わない。最初の反復も完了しなければ合法heuristic fallback。passも1plyとして符号反転し、相手pass後の同色再着手は2回の符号反転で扱う。

TTは盤面/残駒/実得点/手番/passesと探索mode・深さを区別。選択集合をheuristicで固定してからTT手を順序付けし、TTで枝集合が変わることを避ける。終局評価はボーナス込み得点差。小終盤で別実装の全探索と値/最善手を照合し、bound・中断fallback・入力不変性・pass連続手をテストする。

新規開発prefix seed510000/32以上で25ms・100msの旧2ply対新探索を評価、6msは8prefixの補助比較。開発で幅/厳密切替条件を調整した場合は新validation seed520000/32で方式と主要予算を固定し、その後fresh530000/96prefixで同予算旧探索と先後交換比較する。全て既存学習重みを固定。prefix単位CI、実ms/超過率、完了深さ、node、TT hit、厳密解割合を保存。計測中に生成/監査を並行しない。全体2時間を目安に有限run、改善なしでも原因を保存して研究branchへpushする。

Deep development selected config=late; next newvalidation520000.

### 開発途中の観測と追加候補（初回holdoutを見る前に記録）
deep設定は25ms −0.547点、100ms −0.641点、late設定25ms +0.172点。deepは候補幅8へ絞る損失と未完了反復の費用が疑われる。厳密探索は残駒8以下という条件のため3〜6回/約1000手しか発動しない。置けない残駒が多い終盤を除外している可能性がある。
初回runは変更せず完走。追加する場合の仮説は「残駒24以下、両者合法手合計6または10以下だけ厳密探索し、それ以外は旧2plyを使う」。対象rootでは旧同等の全root×上位3応手の2plyを先に完了し、厳密探索が未完了ならその値へfallback。追加学習なし。候補6/10を新開発seed511000/32で25/100ms比較、新validation521000/32で固定、fresh531000/96を最終評価とする。初回holdout530000の結果を設定選択には使わない。
追加候補の最終評価では、選択したhybrid対旧2plyを主比較、対初回validation選択方式（同じ新主要予算）を副比較として同じfresh531000/96prefixで保存する。副比較による再選択はしない。新方式の学習効果ではなく、固定重みでの探索方法の差として扱う。

Deep final selection frozen: {'kind': 'deep', 'budget_ms': 25, 'root_width': 12, 'width': 3, 'deep_gate': 14, 'exact_pieces': 8}, validation mean=0.640625; fresh530000 begins.

### 初回完了
validationでlate/25msを固定。fresh530000/96prefix・192局: 94勝91敗7分、得点差+0.10417、CI[-0.15104,+0.33867]。新18.772ms/手、旧17.329ms/手。選択探索平均完了深さ6.13、厳密解12/3009呼出（試行12/12）。超過9/3009対旧0/3008。安定した改善は未確認。全50tests（独立小終盤13局面を含む）PASS、全対局再生・算術・prefix分離・厳密解12標本の独立再証明PASS。開発段階で事前記録した低mobility厳密探索を別seed群で継続する。初回holdoutを再選択に使わない。

Exact-hybrid fixed aftervalidation: {'kind': 'deep', 'budget_ms': 25, 'root_width': 12, 'width': 3, 'deep_gate': 24, 'exact_pieces': 0, 'exact_mobility': 6, 'only_exact': True, 'max_depth': 2}; fresh531000 begins, earlier method same-budget control={'kind': 'deep', 'budget_ms': 25, 'root_width': 12, 'width': 3, 'deep_gate': 14, 'exact_pieces': 8}.

### 固定重み探索・追加案完了
低mobility閾値6、25msを新validationで固定後、fresh531000/96prefixで旧2plyに90勝89敗13分、+0.21354点 CI[-0.06771,+0.51042]。初回反復深化との同予算副比較は89勝92敗11分、−0.203125点 CI[-0.65625,+0.16146]。厳密解96/2958呼出、98試行中96完了。旧2plyを安定して上回る改善は未確認、既存推奨維持。初回・追加の各holdoutで再選択なし。
全53テスト7.04秒PASS、CLI CUDA対局終局PASS。両runの全対局再生・算術・CI・時間・prefix分離・選択・固定重みhash監査PASS、独立全探索で13小終盤＋各run12証明標本の値/最善手一致。run合計1376局/1203.17秒、GPU学習0更新、追加checkpointなし。READMEとartifacts/exact-hybrid/conclusions.mdへ全結果・実ms/超過・限界・次案を記録。mainは変更せず研究branchで提出する。

## 応手脱落の固定node診断（基点f65bdbf）
追加学習なし、既存value/alpha=.5/gate24を固定。新独立16ゲーム(seed620000系列、heuristic上位8から確率的選択)から各1中盤と1小終盤を採る。独立標本単位はゲーム。小終盤は全合法終局探索、中盤はroot上位24/相手上位12/以降幅6・4ply・上限50000nodeの参照（厳密解ではない）。全root/応手の参照値・浅い葉値を保存し、応手上位3による最悪応手脱落、root幅による最善候補脱落、選択regret、葉値誤差を分離する。小終盤は別実装全探索と照合。
再現可能な固定2048nodeの幅(root6/24、reply3/12)×深さ(2/4)診断も実行。nodeは訪問状態/評価leaf、期限なし、全root値が完了した候補だけ採用、上限到達時の未完了候補は破棄。各局面・条件でcacheなし。root候補集合外の真の中盤最善手は保証しない。
診断後に根拠を記録し変更を1つだけ実装する。新validation seed621000/32prefixで設定固定、fresh622000/96prefixで先後交換・共通25ms。主評価は両者とも試合ごとにTT/適応コスト履歴をリセット、phase間も新instance。旧warm-cache成果は変更しない。時間計測と監査を同時実行しない。全体90分目安、有限runで検証・commit/push・正式finish。

### 診断結果と実装前の選択根拠
16独立ゲーム各1中盤/1終盤を完走。中盤参照(選択4ply)に対し、候補root平均48.18%で最悪応手が上位3外、過大評価gap平均0.342点。root6制限の損失0.250点、浅い葉MAE1.131点。全24root/3応手/2plyのregret0.592点、12応手へ拡大して0.581点。幅拡大単独の信号は弱く、葉評価誤差の寄与が疑われる。一方4ply/幅3は0.310点、4ply/幅12は2048node制約でroot完了減のため0.340点。終盤は応手脱落0、旧regret1.0点、4plyで0.25点なので深さ/評価の問題も分離できた。
今回の変更は「2plyの相手応手幅だけ3から6または12へ拡大」。root順序、value、scheduler、予算は保持。新validationで6/12を比較し得点差最大の幅を固定する。単独拡幅で改善しない場合は、候補脱落の存在だけでは棋力改善へ直結せず葉評価/計算配分が支配的という仮説を次案に残す。これ以外の探索変更は加えない。

Reply width fixed to 6 after newvalidation621000; fresh622000/96 begins.

### 応手幅診断・最終結果
validation621000で幅6 +0.78125点、幅12 −0.21875点のため幅6を選択。fresh622000/96prefix/192局は89勝98敗5分、−0.23958点 CI[-1.14596,+0.61471]。途中返答の「幅12を選択」は誤記として訂正済み。selection・対戦・監査は一貫して幅6。旧holdout再選択なし。
実17.726/17.541ms、超過1/2983対0/2990、完了root20.53/21.68（pass含む）。改善未確認、推奨維持。中盤は上位3の応手脱落48.18%だが、単純拡幅のregret差は僅か。4ply幅12は固定2048nodeで完了root3へ減り、葉評価と配分の問題が残った。次は候補別終局rolloutにsequential halvingで配分する別探索系統を、小規模固定node診断から検討する案。
全60tests 7.05秒、16終盤独立全探索、32条件固定node反復再現、全対局/算術/時間/完了候補数/選択/hash分離監査PASS（監査5.83秒）。CLI CUDA対局終局PASS。実戦320局175.21秒、GPU追加学習0。詳細と再現手順をartifacts/reply-diagnosticsへ保存。main・旧warm-cache成果は変更せず研究branchへ提出する。

## 終局rollout配分（基点376dce2、事前計画）
追加学習なし。root候補8: heuristic上位8、または上位4に異なる駒/盤面象限の候補を加える多様化案。continuationは両者heuristic上位3から一様ランダム、終局得点差のみ。共通乱数は候補間で同じsample seedを用い、continuationの各plyでpassでも1回消費する。同seedでも盤面/合法集合が違うため同じ手を意味しない。平均rolloutは戦略的な相手の適応・最悪応手を表さない。
均等配分32完走対sequential halving32完走（8候補×2、4候補×2、2候補×4）。deadline時は完了した全active候補roundだけを採用、最初の全候補1round未完了ならheuristic先頭fallback。中断/期限後完走は評価へ入れない。完了round内のみ同数標本で比較し、stage途中の追加標本は選択へ反映しない。
新規独立12ゲーム(seed720000+)各中盤1・小終盤1。固定seed/完走数で配分・候補集合の診断、別seed64完走/候補の平均を中盤参照（厳密解でない）、小終盤は独立全探索。各方式3sample seed、同ゲームroot/seedを独立ゲーム数へ水増ししない。25/100msのpilotは均等/適応を同じ局面で計測、旧2plyも同予算で測定。GPU/CPU他計測・監査を重ねない。
継続基準: 100msでも中盤の80%以上で全8候補最低1標本が揃わない、または中盤完走数中央値が16未満なら現実装は配分比較に不足と判断し、大量validation/holdoutを省略して原因を返す。基準を満たす場合のみ新validation721000/32で方式/25または100msを固定しfresh722000/96prefixへ進む。固定予算同士の旧2ply対照、cold reset、実ms・超過・完走・fallbackを保存。全体90分目安。

### pilotの回復・配分テストでの修正
初回の独立終盤oracleは1局面で20000node上限に達した。部分結果を保全して再開し、その局面は厳密解と主張しない。加えて配分数テストが、実装のround切捨てにより予定16+8+8ではなく8+12+12になっている点を検出。開始直後の実戦runを停止し、全候補の最初2標本を優先する16+8+8へ修正、7テストPASS。修正前pilot/途中資料はpre-fix-pilotへ隔離し、最終選択・比較には利用しない。正しい配分でpilot全体を再実行する。

Rollout final selection {'method': 'halving', 'budget_ms': 100, 'diverse': False}; fresh722000/96 begins.

### 終局rollout最終結果
修正後devで上位8候補、100msを選択。validationはuniform−1.71875点、halving−1.6875点のためhalvingを固定。fresh722000/96prefix・先後交換では旧100ms2plyに71勝113敗8分、−3.03125点 CI[-4.515625,-1.546875]。halving対uniformは101勝84敗7分、+1.125点 CI[0.375,1.890625]。配分効果と全体の劣位を分け、既存推奨維持。holdout再選択なし。
主比較実73.276ms対55.417ms、超過1719/3003（最大100.970ms）、fallback472/3003。序盤（残駒24超）のfallbackは472/1342。主比較個別完走51577、採用46267、部分round破棄5310。平均相手方策バイアス、root8制限、序盤fallbackの寄与は未分離。次案は旧2plyと候補/序盤条件を揃え、相手minを扱うtree searchへ配分を移す固定simulation対照。
全68tests7.44秒、独立終盤10/12（2局面はnode上限・厳密値非主張）、3189終局trace・159ラベル再生成・96固定配分再現・512局全再生/算術/時間/選択/hash分離PASS。監査側のsnapshot共有listによる破壊的再生をcloneへ修正し、保存snapshot不変の回帰テストを追加後、全監査を再実行してPASS。実対戦・保存ラベルの生成コードはこの監査修正で変更していない。
修正後pilot127.94秒、本対戦1073.90秒、追加学習0、checkpoint作成0。CLI終局PASS（-67/-5）。READMEとartifacts/terminal-rollouts/conclusions.mdに制約・計算量・次案を記録。mainと旧成果は保持、研究branchのみ提出する。

## Minimax refinement — 2026-11-05 事前計画
基点eb7dbd4、branch research/minimax-refinement。学習なし、value固定。上位128root/相手上位3/初期2plyを保持し、全初期候補完了時だけ余剰時間をdepth4（内部幅3）のroot単位追加negamaxへ配分。相手はmin、自分はmax、葉は手番視点。枝限定探索であり完全minimaxとは呼ばない。未完了rootは初期値を保持、未評価rootは選択対象外。比較は追加なし(control)、初期値の高い順(priority)、元のheuristic順(uniform: 各root一回)で同一初期処理。初期候補数が時間で制限された場合は追加探索しない。予測費用1.5倍と最低1ms/予算5%余裕を確保。coldは毎試合、TTなし。
新規12独立ゲーム820000+、各中盤1・終盤1。固定node上限で再現性と独立終盤oracleを検証。devで25/100ms比較。validation821000/16prefixで優先配分対旧2ply/配分対照。進行条件: 中盤追加完了率>=50%、validation対旧平均得点差>=-1、95%得点差CI上限>=0、超過率<10%。満たさなければ一度設計を修正し別validation821100で確認。二度目も不適ならholdout省略。進む場合主要予算固定後fresh822000/96prefix先後交換、旧2ply主対照と追加なし対照。時間計測中の他作業負荷なし。途中結果・失敗も保持、全体90分目安。

### 初回pilot
新規12ゲーム・24局面、固定2048node再実行一致、独立小終盤oracleも確認。中盤25msは追加完了0/12、初期完了0/12、平均23.15ms。100msは追加完了12/12、初期完了12/12、平均6root追加、実88.15ms、初期49.13ms、超過0/12。100msを新validation821000/16へ進める。固定nodeの値は幅3限定depth4であり完全解ではない（独立oracle欄のみ厳密）。

### validation固定
100ms priority対旧16勝15敗1分/+1.15625 CI[-1.25,4.125]、対uniform15勝16敗1分/0 CI[-1.84375,1.8125]、対control14勝17敗1分/+1.09375 CI[-1.5625,4.1570]。改善証明ではないが事前進行基準を満たす。主比較500手超過0、初期fallback0。仕様修正は不要、reserve variantは未使用。selection.json固定後fresh822000/96prefixへ進行。validationの継承stats.sectionsは未使用ゼロ値のため詳細内訳はdecisions.sections_msを用いる。holdout前に集計表示だけ修正（探索処理変更なし）。

### fresh主対照（監査前）
822000/96prefix・192局、priority100ms対旧100ms:114勝68敗10分、+2.47917点 CI[1.42161,3.61979]。旧holdoutの再選択なし。追加なし対照を継続、main推奨は維持。

### 最終結果・検証
fresh96prefix/192局の同初期処理control対照も111勝70敗11分/+2.48958 CI[1.40104,3.625]。共通100ms下の追加探索には正の差、priority配分そのもののuniformに対する優位は未確認。主対照実70.679/旧51.612ms、追加13319/13320root、初期fallback0/3001、超過0/3001、最大95.091ms。実消費時間同一や学習重み改善とは主張しない。main推奨維持。
全74tests7.46秒PASS、480対局全再生/得点/独立CI算術/時間/hash監査PASS、独立小終盤12件再照合PASS、保存状態hashのvalidation/holdout重複0。CLI play完走(-68,-7)、help確認。RTX5090・既存CUDA value実推論、学習0更新・checkpoint新規0。pilot27.657秒、validation192.367秒、holdout727.266秒、主要実験計947.291秒。実装込みレポート時26.4分。時間計測中の監査・テスト重畳なし。
README/METHOD/conclusions/run-manifest/verificationへ設計・条件・全成績・計算量・再現コマンドを保存。JSONは大きな2ファイルのみ空白圧縮、parse同一確認。次案は実対戦で改善を示した探索を教師とする低予算policy学習の小規模検証（学習成功は未検証）。今回は追加開発せず秘書へ判断を返す。

## Root policy蒸留 — 2026-11-05 事前計画
基点12a9963、branch research/root-policy-distillation。まず固定priority100ms/幅128・内部3/value alpha=.5 gate24をfresh seed920000/96prefix先後交換で旧100ms2plyへ再確認。設定は結果で変更しない。同じ実装の独立標本再確認であり独立実装監査とは呼ばない。時間対局中は収集・学習・監査を実行しない。
収集は新規ゲーム単位にtrain/dev/validation分割。root候補は従来heuristic上位128に固定、相手応手は従来上位3のまま。教師の選択・初期値・追加完了root/深度・値差・候補hashを保存。学習は教師の最終選択の分類だけ（教師の判断を再現するラベルであって厳密最善手ではない）、値回帰はせず、未評価候補はloss maskから除外。深度混在値の差を正解順位として使わない。教師に追加探索完了がある局面を主学習とし、初期2plyだけの局面は保存するが学習ラベルには混ぜない。
学生は軽量root特徴+state MLP、候補ごとの順位のみ変更。初期モデル・データ共有、更新seedは901/902/903（shuffle等のみ独立）。同候補集合/固定value/2ply/相手3でheuristic順・未学習順・学習順を比較。GPU同期/推論/合法生成を手番予算に含む。
pilotはまず約300独立ゲーム、各最大4局面。学習信号基準: dev top1が未学習より5ポイント以上改善、top5も改善、25msで順位推論中央値<2.5ms。満たさなければ大規模化せず停止理由を保存。達成時3seedを同程度更新予算で学習、新validation921000/16prefixで6/25ms予算とcheckpointを選択。validation対旧平均得点差>=-1かつCI上端>=0なら選択固定後fresh922000/96prefixを先後交換、旧value/2ply対照。原則各seedを評価しseed選択による過大評価を避ける。低予算推奨維持、追加GPU主学習上限30分、全体2時間。

### 教師の独立標本再確認
fresh920000/96prefix・192局:98勝89敗5分、平均+1.40625点 CI[0.23424,2.625]。point-rate .52344 CI[.47917,.57031]。得点差は正、勝率の明確な優位は未確認。設定調整なし。同じ実装であり独立実装監査ではない。時間計測終了後、root蒸留収集へ進む。候補特徴39（既存12+駒one-hot21+位置/幅6）とstate49。未学習はheuristic prior+zero residualで同順位。3seed×3000更新、1000/3000のcheckpointをvalidation比較候補とし、seedは選ばず共通予算/stepを選択して全3seedを報告する。

### pilot・停止基準の扱い（holdout前）
収集300game/1192root、追加完了587、同深度比較可能571（train463/dev59/validation49）。採用rootのsplit間hash重複0、未採用rootには7種類のsplit間重複あり。初回品質assertが全収集rootへ広すぎて失敗したが、検査範囲を確認し学習採用データに漏洩なしを確認（重複を隠さずauditへ保存）。3seed×3000更新のGPU主学習計8.047秒。3000更新でdev悪化、1000更新の平均top1は31.64%（未学習28.81%）、top5は68.93%（67.80%）。事前5pt目安に不達。推論中央値は約0.28msで費用基準は達成。
小規模実戦validationでは25ms/1000が全3seed正（+0.59375,+0.65625,+2.21875）、平均+1.15625 CI[-1.84401,3.84375]。6ms/1000 -0.19792、6ms/3000 -2、25ms/3000 -0.05208。このためデータ増量・更新追加は停止し、元依頼の「一致率だけで採用しない」に従い実戦信号を優先して既定のfresh評価のみ実施する。これは5ptを必須とした事前簡易gateからの変更であり、基準達成とは記載しない。holdout未参照の時点で判断を記録。checkpointは全seed1000、25msを固定、seed選択なし。以後設定/学習を変更しない。

### 最終結果
教師再確認は+1.406点 CI[0.234,2.625]。学生fresh922000/96prefix×先後×3seed、対旧25msは261勝276敗39分/+0.203125 CI[-0.831771,1.324696]、対未学習順位は269勝270敗37分/+0.321181 CI[-0.821267,1.526128]。3seed/先後をprefix内で平均するCIで独立標本の水増しなし。改善未確認、教師の成績を学生へ流用せず、既存低予算推奨・main維持。
主比較実17.592ms対旧17.501ms。学生超過3/8940手、初期fallback0。data collection132.763秒、教師再確認356.562秒、GPU主学習8.047秒、validation181.946秒、holdout629.540秒。主要処理計1308.858秒、レポート時実経過38.8分。学習の無根拠な延長・追加データ生成なし。
教師trace24局面を完了深度どおりに再計算、1192root候補/mask監査、300生成ゲームと1856評価対局の再生・算術・選択hash監査PASS。全78tests7.30秒。CUDA別process40対20+20、model/optimizer/Python/NumPy/Torch/CUDA RNG bit一致。shared初期weightsも3seed bit一致。CLI完走(-68,-8)。
原因診断はdev30game59rootのみ、holdout再選択なし。25msでteacher手の探索対象包含100%でも最終一致約44%、teacherが初期2ply最大から反転30/59。順序だけで固定2ply最終値の誤差を直せない場合がある。3000更新でtrain改善/dev悪化、same-depth同値候補複数133/571というラベル曖昧さも記録。次案はroot値補正を最終選択へ渡す学習interfaceの上限診断から。現runでは変更せず秘書判断へ返す。
README、METHOD、conclusions、全checkpoint/config/log/raw data、compute-summary、verificationを保存。大きなJSONは空白圧縮のみ、parse同一確認。研究branchのみcommit/push予定、main未統合。

## root-value-correction 着手
d3c4698から研究branch作成。main eb96a90への教師統合は別worktreeで依存を限定して行う。補正学習の段階/gate/新seedは artifacts/root-value-correction/METHOD.md に事前保存。最終root値に補正を加え、2ply上書きを防ぐ。ゼロ補正推論対照を必須とする。

### 教師統合と補正pilot
mainへcbe3b91を通常fast-forward push済み（53 testsと教師/旧2plyのCUDA CLI完走）。研究枝はmainへ入れない。
新24ゲーム48局面の上限診断は同候補mask内で28局面のdepth4選択損失を改善、平均0.753→0。ただし教師情報を直接使った限定集合上の診断であり、棋力や低予算での実現を示さない。事前基準10%を満たしたため240ゲームへ進行。
240ゲームの477局面（重複1を除去）、train381/dev48/validation48。初期値上位4と残候補一様2を同depth4で完了評価。GPU3seed各3000更新を完了。別process40連続 vs20+保存再開40のmodel/optimizer/Python/NumPy/Torch/CUDA RNG bit一致。devの補正選択損失には改善傾向があるが、更新増で予測誤差が悪化するため予定外の増量はしない。

### validation選択固定（holdout前）
新seed1021000/16prefixの15比較480局完了。3seed共通step3000/alpha0.5/25msを平均得点差最大で固定。旧探索へ+1.7708（prefix CI[-0.5734,4.0836]）、同じモデル推論を実行するゼロ補正へ+2.2396（[-0.0208,4.4065]）。平均推論0.4003ms。事前gateの5条件すべてtrue、例外変更なし。推奨変更はしない。新seed1022000/96prefixへ進む。選択ハッシュはselection.json、全結果はvalidation-results.json。主学習/生成/テスト/監査は停止し対局単独計測。

### 最終結果・監査
fresh96prefix×先後×3seed×2対照=1152局完走。対旧は254勝293敗29分、平均+0.03646 CI[-1.00525,1.03129]。対zeroは259勝291敗26分、-0.05729 CI[-1.03819,0.91150]。改善なし、既存value/2ply推奨維持。設定再選択なし。
82 tests、CUDA別process再開、3seed初期重みbit一致。源240ゲーム/477局面/教師24局面再計算/評価1632局再生・算術監査PASS。全源状態と評価状態hash重複0。devでは教師最良候補の評価範囲coverage100%ながら教師保持50–54%、未使用ゲームのMSE/選択regretはzeroより悪化し、汎化が主な課題。結果・次案・再現CLIはconclusions.mdとMETHOD.md。追加学習せず研究branchへ保存する。

## correction-game-scaling 着手
c46a180から新branch。学習/実戦の偶然の最大値選択を止め、固定評価ゲームで240/960/2400訓練ゲームの曲線を先に検証する。3fold×3更新seed、正則化2条件、1000更新gate固定。詳細と打切り基準はartifacts/correction-game-scaling/METHOD.md。旧再開の正確な比較は40連続 vs20+追加20（総40）であり、前表記を訂正する。

### 240ゲーム条件
生成720源ゲーム（bank360+pool360）、有効root1347。実訓練237–239ゲーム/fold、固定評価117–120ゲーム/fold。3fold×3seed×2条件の各3000更新完走。1000更新時plain MSE9.418対zero7.304、regret0.53193対zero0.53711。decayもMSE9.562で悪化。対戦gateは未達。960ゲームへの予定比較を進める。
84 tests PASS。新optimizer2条件とも別process40連続 vs20+追加20=総40のmodel/optimizer/全RNG bit一致。旧表記は通算目標stepの曖昧な書き方だったと確認。

### 960結果と追加生成だけの例外（2400生成前）
1000更新のMSEはplain9.418→7.967、decay9.562→7.954と改善。一方regretはplain0.53193→0.54018、decay0.53736→0.54429で非改善（zero0.53711）。対戦gateはすべて未達。
独自の追加生成gate「両指標非悪化」は未達のままextension-gate.jsonに保存した。ユーザーの早期打切り許可は『学習曲線が悪化のみ』であり、今回はMSE改善があるため、その条件に当たらない。計算予算にも余裕がある。独自gateが依頼の条件より厳しすぎたため、追加生成のみの例外をextension-exception.jsonに事前記録し、依頼された2400点まで測る。対戦の1000更新・全fold5%・CI gateは変更せず、validationの勝ちで覆すこともしない。

### 学習曲線完了・対戦gate未達
2400条件まで54run/162000更新を完走、GPU主学習176.534秒。1000更新plainはMSE6.92786（zero7.30425）、regret0.522586（zero0.537114）。decayは6.90906/0.523928。平均改善傾向はあるが、両条件ともfold1はMSE/選択損失悪化、fold2も選択損失悪化、CIに0を含む。固定した対戦gate未達で、validation/holdoutを作らず終了。生成gate例外は記録済み、対戦gate変更なし。
85 tests、2条件CUDA別process40 vs20+追加20=総40、3960源ゲーム全再生、7424root入力/候補/mask、入れ子配列・分割hash・CI/gate算術、教師75root+偏り診断36root再計算PASS。源ゲームは完全手順hashも3960unique。候補入力衝突は1037root、異なる教師値を持つ衝突群59件だが観測MSE下限0.000349で主因と断定しない。
学習曲線PNG/SVG、日本語conclusions.md、再現METHOD.md保存。旧再開表記を正確に訂正。main変更なし、研究branchへcommit/pushする。

## candidate-mask 着手
# 候補配置入力の対照実験（事前固定）
基点312e9dc。main変更なし。既存3600ゲームpoolのみ再利用し、各fold訓練は2400 nominal games。過去bank/holdoutを学習・選択に使わない。新bank360ゲームID200000–200359、seed1330000–1330359。pool/seat/fold層化、45/65%進行root、ゲーム/局面hash分離。教師depth4-depth2、固定value、width3、top4+random2ラベル、Huber、alpha .5は保持。教師値は真のゲーム価値ではない。
2表現: summary=従来6ch盤面CNN+39候補要約、cells=候補の占有セルを7ch目に追加して候補ごとCNN。共通重みは同一初期値、追加channelのみ新規。両者1000更新、AdamW lr.001 weight_decay0、batch32 root、既知候補のみ損失。3fold×更新seed1101/1102/1103。ゲーム分割/データ/初期共通部分/ミニバッチ乱数は共有。追加正則化・サイズ・checkpoint探索なし。
進行gate（前回と同じ、不変）: 各foldの3seed平均でraw MSEとalpha .5選択損失の双方がゼロ補正より5%以上改善、各seedのfold平均選択損失がゼロより改善、ゲーム単位でseed平均後bootstrap4000の両改善95%CI下限>0。1000更新のみ判定。失敗なら対戦なし。raw MSEにはalphaを掛けない。条件を途中変更しない。
訓練GPU主計算合計1200秒以下、今回90分目安。通過時のみ新validation32prefix、その後96prefix先後交換25msへ進む（全体2h）。入力衝突回帰、勾配、CUDA別process40 vs20+20、教師再生、分割/算術監査、候補batchメモリ/遅延を記録する。


## candidate-mask 診断完了
候補配置の区別・勾配確認、18run/18000更新完走。新bank356ゲーム672roots。summary/cells raw MSE6.45957/6.45132、regret .520514/.537984、zero .540837。両gate未達、対戦を実施しない。89tests、CUDA別process40 vs20+20、教師75root再生、split/算術監査PASS。全履歴には共通状態あり、採用ラベルrootの重複は0（詳細REPORT）。GPU主学習19.82秒。main/推奨維持。root補正の微調整を区切る判断を推奨。
# Terminal PPO 事前計画
基点f6db8c9、新branch research/terminal-ppo。main変更なし。root補正は終了。
全合法手（passは合法手なし時のみ）の可変長分布。盤面6ch CNN、残駒/score/pass49状態、候補39要約と正確な占有セルの位置埋込・盤面conv局所特徴を使用。maskでpaddingを除外。候補上位制限なし。ネット単体、探索なし。
初期policyは従来heuristic logitへのゼロ残差、valueゼロ。初期seed1400共有、学習seed1401/1402/1403（対局乱数・sample・minibatchは別）。初期checkpointを保持。同一アーキテクチャ学習前を対照。
終局勝敗+1/0/-1のみをMC return（gamma1）とする。各記録の実プレイヤー視点へ終局scoreから直接割当、advantage=return-old_value、batch内標準化。得点差は評価指標のみ、shapingなし。PPO clip .2、value MSE係数.5、entropy係数.01、Adam lr .0003、grad clip .5、2epochs、minibatch16states、近似KL .03超でepoch中断。
16並列対局を1収集単位とし、current同士50%/凍結過去policy25%/heuristic25%を毎単位固定。後2種では座席を均等化しcurrent側のみ更新、current同士は両側記録。凍結過去は初期から開始し4単位毎に更新する1スロットpool（過去全履歴ではない）。毎対局新seedの合法random2–6ply prefix。全局終局まで完了後更新、保存は単位境界。iteration=収集+更新。
まずsmokeとCUDA別process再開。pilot各seed64iterations=1024games、保存16/32/64。3seed合計生成+更新wall上限1800秒（各seed600秒、途中の単位は終了して記録）。上限時は未達と明記。timingはCPUlegal/pack、GPU推論同期、更新、残余を記録。途中ckptは3個以下。
validation固定新prefix seed1430000、32prefix先後交換。相手poolはheuristic/各seed学習前を等重み。各seedのcheckpointは勝敗点率平均最大、同値は得点差→早いiteration。oldvalue/2ply6msは選択後の強い比較基準、選択に使わない。
対戦進行gate: 選択後各seedについて両基準への点率>=.5、3seed平均点率>=.53かつ平均得点差>0、さらに各基準へのprefix単位seed平均bootstrap95%下限点率>.5。旧教師MSE gateは使わない。未達なら最終holdoutは作らない。診断で根拠があれば1回だけ修正・新validationで再pilot可、その条件は開始前に追記する。
gate通過時のみ設定固定、fresh seed1440000 96prefix先後交換で3seed/学習前/heuristic/旧valueと比較。時間同等を主張しない。単独計測、試合cold。全対局再生・終局報酬・算術監査。PPO参考 https://arxiv.org/abs/1707.06347 。初期v1は46状態+12要約MLP・単純終局得点差actor-critic、今回は空間/正確配置・勝敗PPO・batch/pool・clip/固定検証を導入。
# 1回だけの再pilot（初回結果確認後・実行前固定）
初回64iterations×3seedのgateは未達（heuristic点率.5234、CI[.4635,.5833]）。損失有限・KL小・entropy約2.6、方策崩壊なし。経験量が1024games/seed、生成+更新計約112秒と小さいため、変更軸を経験量だけに限定して検証する。
常に各seedの最新64iterationから継続（validation選択checkpointからは開始しない）。モデル/初期重み/損失/PPO係数/相手割合/学習率は不変。総512iterations=8192games/seed、保存128/256/512、3seed生成+更新の累積wall上限1800秒を維持。訓練prefixは通算iterationの新seed、旧validation/holdoutは使わない。
出力artifacts/terminal-ppo/extended。新validation seed1470000、32prefix先後交換、固定heuristic/学習前poolでcheckpointを選ぶ。同一gate。通過時だけfresh holdout seed1480000、96prefix。未達ならこの依頼での修正は終了し、追加ハイパーパラメータ探索をしない。
保存実装のみ大きな履歴重複を避けてgzip圧縮（mtime0）、学習意味は不変。新保存形式でもCUDA別process再開を再検証する。

## terminal-ppo 初回pilotと再pilot進行
初回3seed各1024games完走、生成+更新wall合計111.68秒。固定32prefix validation選択後は対heuristic点率.52344/得点差+.89583、CI点率[.46354,.58333]でgate未達。旧value6msに各seed負け。3072訓練対局/1344評価対局再生、69348終局target、分割・算術監査PASS。CUDA別process2 vs1+1もPASS。
1回の修正として経験量のみ8192games/seedへ増加、最新64iterationから継続、新validation1470000に変更。設定・gateはREPAIR-PLANで事前固定。2seedは約305/307秒（初回を含む累積）で完走、3seed目実行中。pool50/25/25はゲーム比率で、current同士両側学習によりstate比率は概ね2/3,1/6,1/6。実数を監査へ保存する。

## terminal-ppo 再pilot validation gate通過・holdout開始
3seed各8192games完走。新validation1470000/32prefix先後交換で点率平均.60417、CI[.53900,.671875]、平均得点差+1.52083、CI[.31771,2.78125]（学習前とheuristic双方）。全seed点率>=.5で事前gate通過。選択は1401=iteration128、1402=512、1403=512。selection.jsonのcheckpoint/hashを固定し、fresh seed1480000/96prefix先後交換を開始。holdoutによる再選択なし。旧value/2ply6msにはvalidationで各seed負け、main推奨維持。

## terminal-ppo 最終検証完了
fresh1480000/96prefix先後交換、3seed平均対heuristic点率.580729（CI[.539041,.622396]）、得点差+1.47222（CI[.775998,2.112891]）。対学習前点率同じ/得点差+1.45660。1402/1403は改善、1401はほぼ互角でCIに0。対旧value6ms点率.296875/得点差-4.75521。新ネットは探索なし、思考時間同等とは主張しない。main/既存推奨維持。
生成+更新累積919.32秒、24576games、70975minibatch更新、556410current records。全93tests PASS、CUDA別process6iterations連続vs4保存+追加2でモデル/過去policy/optimizer/対局/RNG bit一致。初回+再pilot validation各1344局、freshholdout1728局の合法/終局/符号/算術を監査、訓練/validation訪問状態重複0。CLI完走。checkpoint選択固定を維持。REPORT/METHOD/設定/curve/各局結果/再現コマンド保存。必要ckptのみ保持、削減hashをpruned-checkpointsへ保存。

## terminal-ppo holdout分離の訂正（モデル選択変更なし）
追加監査でseed1480000 holdoutの一部prefixが訓練prefixに一致と判明。先のfresh成績は独立性条件を満たさないため最終結論から外し、全ファイルをextended/holdout-1480000-overlapへ保全。算術監査PASSと訓練分離PASSは別であり、前者だけでは不足した。
固定checkpointは一切変更せず、seed1500000の事前生成候補から訓練prefix hash/訓練state hashおよび旧validation/holdout/diagnostic prefixと一致するものを除き、最初の96prefixを固定（holdout-disjoint-plan.json）。新ゲームの結果はフィルタ・モデル選択に使わない。追加学習なし、対戦のみ再実行。旧結果と失敗監査logを保持する。

## terminal-ppo 最終確定（seed1500000）
分離済み96prefix/1728局の最終監査PASS。訓練prefix重複0、訓練訪問状態重複0、固定selection hash一致。対heuristic点率.605035（CI[.569444,.640625]）/得点差+1.600694（CI[1.006944,2.196181]）、対学習前点率.606771/得点差+1.633681。対旧value6ms点率.331597/得点差-3.684028。新holdoutでは3seedとも平均差が正だが、3seedと共有prefixに条件付けた結果。旧1480000結果は最終結論に使わない。分離チェックを評価CLIの実行前へ移し回帰テスト追加、最終94tests PASS。全処理終了、main・既存推奨維持。branch commit/pushへ進む。

## PPO learning curve 着手
# PPO学習量比較 事前計画
基点34cd4e6。main変更なし。モデル・PPO損失・終局勝敗報酬・16ゲーム同期生成・相手割合・4iterationごとの過去policy交換を維持。seed1401/1402/1403の512から1024/2048へ継続。追加生成更新wall上限合計3600秒、各seed1200秒、iteration境界で停止。非有限値は即停止。
前回selected (128/512/512) は固定。新validation seed1530000、32prefix、先後交換。各checkpointを前回同seed selected、heuristic、旧value6msと比較。選択はpointの重み付き平均(previous .5/heuristic .25/oldvalue .25)、同値なら同重み得点差、さらに早いiteration。旧selectedも候補に含めるため退行時には維持できる。
新dev seed1520000、8prefixでgreedy/確率的方策を診断のみ。主評価greedy。選択固定後seed1540000、96prefixを訓練全訪問状態・過去評価開始状態/sequenceから除外して固定。成績によるprefix再選択なし。全訪問状態監査は空盤面等prefix以前と採用評価局面以降を区別。prefix単位bootstrap、seedは同prefix内で平均して集約。
ネット単体と旧探索6msは実消費時間が異なる。学習量のみを変更し探索は付けない。GPU更新時間のみでなく生成・保存を含む追加wallを計測。履歴は外部gzip sidecarとして保存しcheckpoint肥大化を避ける。初期512履歴と継続履歴を監査する。

PPO継続中: seed1401 1712iteration、1024保存済み。直近64iteration訓練内平均勝敗報酬 {'current': -0.0546875, 'past': 0.0078125, 'heuristic': 0.27734375}。強さ評価とは区別。非有限値なし。評価/再開監査は訓練後に直列実行。

seed1401は2048iteration/累積32768games完走、追加998.509秒。seed1402を継続開始。保存形式以外のPPO本体は34cd4e6とbyte一致。

seed1402も2048iteration/32768games完走、追加1019.046秒。2seed合計2017.555秒。seed1403開始、学習条件変更なし。

3seedすべて2048完走。追加wall合計3022.566秒、追加73728games/累積98304games。学習停止しCUDA別process再開・全テスト・全訓練訪問状態監査へ移行。

## PPO learning curve 完了結果
3seedともvalidationで2048選択、以後固定。fresh seed1540000/96prefix/先後交換/1728対局。前回selected比 +7.203 [95%CI +6.562,+7.856]、heuristic比 +6.205 [+5.439,+7.005]、旧value/2ply6ms比 +2.104 [+1.272,+2.891]。seed内の2座席と3seedをprefix内で平均。
訓練98304対局/3306730全訪問hashを再生監査。評価前開始状態・評価後実訪問の重複0、全3776評価対局算術PASS。98tests PASS、CUDA別process 512→518 vs 512→514+4、過去policy交換516を含みmodel/past/optimizer/RNG/履歴bit一致。
追加生成・更新・保存3022.566秒、73728games、213158 minibatch updates。対旧探索の実平均ネット1.215ms/旧探索4.643ms。同時間比較とは呼ばない。main変更なし。
補足devでseed1401 latest512を追加診断（前回selectedは128）。同dev8prefix point greedy .9375/stochastic .5。選択固定後の診断のみで再選択なし。seed1401のlatest512への96prefix直接比較は未実施と明記。
検証・対戦・レポートまでの経過約65.0分。結果/全再現手順: artifacts/ppo-learning-curve/REPORT.md。残りcommit/pushと正式finish。

# 2048再確認・配布と4096継続 事前計画
基点df7bdf8、研究branch research/ppo-4096。mainへは2048推論だけを追加し既存デフォルト維持。例示seed1401を結果より前に固定。seed選び直しなし。
第一段階: seed1570000の新96prefixを旧全訓練訪問状態3306730件/過去評価prefix・開始状態から事前除外。固定2048の3seed vs旧value6ms/固定100ms priority教師、seed1401のみ再開元512も比較。先後交換、prefix CI、全実時間と事後全訪問監査。設定調整なし。
第二段階: policy推論を履歴不要へexport。CPU/CUDA出力一致と合法性、CLI人間/AI対戦、新cloneを検証してmainへ通常cherry-pick。巨大訓練履歴は研究branchのみ。旧valueと100ms教師を残す。
第三段階: 各seed2048のoptimizer/past/RNGから4096まで継続、3072/4096保存。損失/報酬/相手割合/学習率/16ゲーム同期処理変更なし。追加生成更新IO wall合計5400秒、各seed1800秒。非有限で停止。上限到達はiteration境界で保存し無期限延長なし。
継続学習の新validation seed1590000/32prefix、各seed2048/新3072/4096を比較。選択基準は前回と同じ previous2048 point .5 + heuristic .25 + oldvalue6ms .25、同値なら同重み得点差、次に早いiteration。選択固定後seed1600000のfresh96prefixで対2048/旧6msのみ評価。すべて全訓練状態と過去評価prefix除外、先後交換、prefix内でseed平均bootstrap。再確認prefixを訓練の開始状態へ流用しない。生成/更新と時間対戦は直列。

固定2048の再確認1344対局PASS、全訪問重複0。旧6ms得点差+0.696 CI[-0.195,1.590]、point .5608 CI[.5182,.6016]、100ms教師へ−2.026 CI[-3.087,−.988]。seed1401対512は+7.536 CI[6.135,8.938]。main d8b9803へ推論のみ統合push、新clone CPU専用/GPU/人間先後/既存探索対戦PASS、main57tests。研究側104tests/export CPU CUDA bit一致。継続へ移行。

- 完了: 3seedとも2048→4096、追加98,304局/生成更新3980.70秒（66.3分）。validation選択は1401=3072、1402/1403=4096。fresh96prefix先後交換で対固定2048 +3.5625 [2.7430,4.3769]、対旧6ms +2.5625 [1.8263,3.2691]。CIはprefix単位、3モデル固定条件。
- 分離監査: 再確認開始状態の訓練prefix/全訪問重複0。validation 1728局、holdout 1152局の実訪問訓練重複0。選択・得点・報酬・再生監査PASS。CUDA別process 6iteration vs2+4（past交換を含む）bit一致。
- main d8b9803 push済み: 固定2048の3seedを履歴不要の推論モデルとして提供。新cloneのCPU専用環境/人間先後/旧探索、CUDA完走、main57tests PASS。新継続モデルは研究branchのみ。既存デフォルト維持。
- 詳細は artifacts/ppo-4096/REPORT.md。時間条件が異なる100ms教師への新モデル優位は未検証。有限pipelineは約94分、無期限runなし。

## 固定新世代PPO再確認開始
基点 fd527ca。追加学習なし。選択は artifacts/ppo-4096/selection.json の1401=3072/1402=4096/1403=4096を固定し、再選択しない。
新seed1620000から96prefixを生成。全訓練訪問状態（ppo-4096のSHA検証済みindex）と過去評価prefix/開始状態を除外。同一96prefixを3seed×固定2048/旧6ms/教師100msの9組で先後交換（1728局）。評価は単独実行し教師cost/refine予測を試合cold。事前監査→全時間評価→事後再生・分離監査の順。
CIは2座席と3モデルをprefix内平均した96prefix bootstrap、seed別CIも保存。固定モデル条件での再確認であり、独立実装監査ではない。timeは全act処理、探索の6/100ms超過率を記録。ネットにはdeadlineがなく同時間比較ではない。
mainへの新世代既定採用は、対2048でaggregate得点差95%CI上端<0またはpoint CI上端<.5なら見送る。それ以外は既に前段階の独立holdoutで改善を確認した追加世代として提供。seed既定1401、旧2048選択可能、グローバル旧AIデフォルト不変。holdoutによるモデル/seedの選び直し禁止。
少数敗局は各seed×2相手（2048/100ms）の最初の負け1局ずつ、各序盤/中盤/終盤の1局面ずつを記述的に診断。これは反実仮想の因果推定ではなくモデル修正・学習には使わない。

- 固定再確認完了: 全9組1728局、共通96prefix seed1620000。対2048 point .63715/得点差+3.9583 CI[3.2031,4.6650]、旧6ms .63368/+2.9184 [2.0955,3.7049]。100ms教師 .52170/+0.7674 [-0.2366,1.7484]は優越未確定。実時間は教師戦PPO1.414ms/教師69.588ms。教師100ms超過0%、旧6ms超過0.0667%。
- 事後合法性・得点・算術・ペア・SHA・過去prefix/開始状態・訓練全訪問分離監査PASS、実訪問訓練重複0。新モデル固定の再確認であり独立実装監査とは呼ばない。追加学習なし、再選択なし。
- main a96d280 push確認。80a5179を22801aaとして通常cherry-pickし、文書と新clone証拠を追加。PPO専用CLIはgen2既定、旧2048指定可能、seed1401/旧グローバルAIデフォルト維持。3seed×CPU/CUDA×26局面export bit一致。main59tests/CPU専用6tests、CPU/CUDA各13局再生PASS。
- 敗局6件は2prefixのみで強い選択偏りあり。中盤合法手不足と3〜5マス残駒を観察したが原因断定不可。次案は同条件の有限継続を第一候補、過去世代poolや小予算探索は別の対照実験が必要。詳細 artifacts/ppo-generation2/REPORT.md。

## 4096→6144有限継続開始
基点 d29f47a。main gen2は変更しない。3seed最新4096のmodel/optimizer/past/全RNGから継続し、5120/6144を保存。seed1401のgen2基準は3072であり再開元4096と異なる。1402/1403は同一4096。
loss/勝敗終局報酬/モデル/LR .0003/16並行ゲーム/相手比率current .5,past .25,heuristic .25/4iterationごとのpast交換を固定。新しいshaping/探索なし。
既存全評価prefix集合を凍結し、その開始状態を含む新生成prefixを棄却。棄却数と旧評価開始状態への偶然の実訪問重複は別に記録する。旧訓練履歴・state indexはSHA付き参照で保持し、追加分だけ保存する。
生成+更新+保存wallは3seed合計5400秒以内。各seed cap最大1760秒、保存余裕を確保し、全体予算の残りで制限。予算到達なら完了iterationまでを保存し自動延長しない。非有限値は停止。64iteration連続窓でentropy平均<.02かつheuristic対戦point<.10なら崩壊疑いとして停止診断。タイムアウトは有限subprocess timeoutで回収する。
新validation seed1650000/32prefix。候補gen2・latest4096・5120・6144（未完走なら存在する保存checkpointのみ）。選択criterionはgen2 .5 + oldvalue6ms .25 + teacher100ms .25 のpoint、同値時同重み得点差、さらに早いiteration。再開元4096への直接対戦は全候補で別記しcriterionに追加しない。gen2自体は必ず候補に残す。seed選別なし。
選択固定後holdout seed1660000/96prefix・先後交換。主比較はgen2とteacher100ms。seed1401は選択と再開元の違いを明確にするため、成績に依存せずfresh同96prefixでlatest4096への補足を常に行う（他seedはgen2=latestなので重複対戦しない）。全seed/個別のpoint・得点差CIはprefix単位bootstrap。旧holdout成績を選択に使わない。
訓練全訪問・過去評価prefix/開始状態を新bankから事前除外。実訪問重複を事後監査。時間対戦は学習/監査と順次実行し、教師/旧探索を毎試合cold。ネット単体で全合法手、推論全費用を計測。同時間比較ではない。

- 事前テスト109件PASS。CUDA別process 4096→4102の6iteration vs2+保存再開4（past交換4100）model/past/optimizer/Python/NumPy/Torch/CUDA RNG/履歴bit一致。既存評価prefix2434件凍結。
- seed1401は6144完走、追加32768局/1314.01秒。seed1402は5120保存済みで継続中。崩壊/非有限停止なし。生成更新と時間評価は順次実行。

- seed1402も6144完走（process wall1282.82秒）。seed1401/1402＋CUDA再開smoke計2634.98秒。seed1403開始。5120/6144は保持し、main/モデル条件に変更なし。

- 全3seedの5120保存完了。seed1403は6144に向け継続中、非有限/崩壊停止なし。学習曲線と各phase費用はseed別curve.jsonに16iterationごとに更新。

- 3seedとも6144完走、5120/6144保存。追加32768×3=98304局。process wallは1401=1315.55、1402=1282.82、1403=1269.89秒、CUDA再開smoke36.62秒込み3904.86秒（65.08分、90分以内）。追加学習終了、全訓練履歴の報酬/合法性/分離監査を実行中。

- 追加分離監査がplay(validate=True)のpassで独立全配置checkerを重複実行し遅延したため、その監査processのみ停止。全履歴合法性/報酬監査とstate indexは完了済み。checkpoint/hash照合後、分離用再生のみvalidate=Falseへ修正して再開する。学習・ルール・評価条件の変更なし、既存監査結果を破棄せず保存。

- 重複pass総当たりを除いた分離用再生が完了。新prefix棄却は1401=84/1402=75/1403=63、prefix混入0、偶然の実訪問旧評価開始状態重複0。元の全履歴合法性/報酬監査は保持。新validation seed1650000/32prefixを開始（生成/学習/監査と分離）。

- 新validation完了。事前criterionによる選択は1401=5120、1402=5120、1403=6144。gen2自体とlatest4096も候補に保持した上で確定。fresh seed1660000/96prefixへ進み、gen2/教師100ms＋seed1401のみ再開元4096を比較。以後checkpoint/seed再選択なし。

- 2026-11-06 明示再開許可、新依頼5d5e572d-61ab-4a04-9964-44e853c01702で復旧。complete=true、学習・2176 validation/1344 holdout局完了を確認。停止記録を再開へ更新。学習/時間対戦を繰り返さずcheckpointロード・hash・監査と報告を仕上げる。

- 復旧検証完了: 6checkpoint CPU/CUDA forward一致、15履歴SHA、selection SHA PASS。全109tests PASS。既存CUDA別process再開6=2+4（past交換4100込み）PASSを保持。2176 validation/1344 holdout再生算術分離監査を再実施し旧JSONと一致、学習/時間対戦の再実行なし。
- 最終: 選択5120/5120/6144。対gen2 point.51476 CI[.48175,.54861]、得点差+.6979 CI[-.0417,1.4254]で追加改善未確定。対教師100ms point.59549 CI[.56163,.63021]、得点差+2.0833 CI[1.3420,2.8768]。1401再開元4096対照も未確定。main gen2維持。有限追加98304局、3process3868.25秒、再開smoke込み3904.86秒。REPORT/compute/学習曲線/復旧ログ保存。次案は凍結複数世代を含むpast枠対照、今回は実行しない。

- 新依頼23150245、research/ppo-generation-poolをcb74785から作成。PLAN.mdにA/B割合、同seed世代固定、外生prefix乱数共有、等iteration/ゲームと実更新数差の扱い、validation/gate/fresh評価を事前固定。main維持。追加1024iteration×6runを上限とし自動延長しない。
- 111tests PASS、B pool循環6144→6150の別process CUDA連続6 vs2+4再開PASS（past交換6148、model/past/optimizer/RNG/pool/履歴bit一致）。2562既存評価prefixを固定除外。6runを直列開始。評価コードはA/B新履歴だけ再生し旧全訪問indexをSHA参照、和集合で分離する。
- A/1401が7168完走、追加16384局、内部wall645.40秒。B/1401を同じ6144から開始。main変更なし、評価は全6run終了後に直列実施。
- seed1401 A/Bとも7168完走。A内部645.40秒、B約670.6秒、各追加16384局。生成/更新コスト差は保持。seed1402 Aへ進行、評価対戦未実施。
- A/1402も7168完走（628.22秒）、B/1402開始。3/6run完了、いずれも追加16384局。early-KLによる実minibatch差を保持して集計する。固定モデル・main・事前gateに変更なし。
- seed1402 A/Bとも7168完走（A628.22秒、B665.01秒）。4/6run終了、追加計65536局。seed1403 A開始。条件変更・seed除外なし。
- A/1403が7168完走（634.18秒）。5/6run終了、最後のB/1403を開始。全run等追加1024iterationのまま。学習終了後にA/B全履歴和集合の分離監査と新validationを行う。
- 全6runが7168完走（各追加1024iteration/16384局、合計98304局）。学習終了、全条件共通7168でselectionを固定。新規全対局再生・報酬・learner対象・pool割当・外生prefix共有を監査中。以後モデル/seed再選択なし。
- 全6run学習wall3895.823秒（64.93分）で終了。新規98304局の合法性/終局報酬/learner対象/pool割当監査PASS。A/B各seedの16384共有prefix列一致、過去評価開始状態の偶然再訪0。A/B全訪問和集合で分離した新validation1670000/32prefix開始。モデル7168固定、時間対戦は単独実行。
- validation27組1728局完了。事前gate通過: B対A point .5755208、固定4相手の平均point B−A=-.02734375（閾値-.05）。固定相手全般でBが優位とは言えない。再生算術分離監査後、設定変更なしでfresh1680000/96prefix・B対A/gen2/教師へ進む。
- fresh1680000/96prefix、B対A/gen2/教師9組1728局完了。B対A point.4826389 CI[.4383464,.5269097]、得点差+.100694 CI[-.765799,.928863]でpool効果未確認。seed1402は退行。対gen2 point.5529514/差+1.621528、対教師point.5711806/差+1.758681の平均成績をpool効果と混同しない。main維持、追加延長なし。
- validation1728/holdout1728局の再生・報酬/算術・対照指定・gate再計算・分離監査PASS。採用開始/実訪問の訓練重複0。全112tests PASS、CLI1局完走（評価には不算入）。REPORT/compute/曲線/再現CLI保存。学習wall3895.82秒、主実験完了5296.26秒。準備関数由来の未使用旧lock項目は元データを保持しmetadata-notesへ明記。
- 新依頼2d02d0e6、research/ppo-short-searchを952d4cbから開始。公開gen2全3seed固定、学習0。PLAN.mdにbatched2ply、6ms4x2/25ms8x3、value勝敗単位/手番視点、完了候補集合fallback、validation/gateと一度だけmargin保守修正を事前記録。main変更なし。
- ppo-short-search: 独立小終盤/視点/pass/fallbackの8テスト追加、全120PASS。dev36ゲーム105rootでvalue校正・CPU/GPU費用を測定、validation前に費用予測を固定。初版25msは対単体point .435/差−2.484で悪化、6msはfallback61%。登録済みmargin .15修正も6ms point .513/差+.245（CIは無改善含む）/fallback61.74%、25ms .458/−.906、両方gate未達。96prefix holdout/教師100ms比較には進まない。追加学習0、有限960対局96.35秒、main a96d280維持。
- 960局の合法再生/得点/paired CI/時間算術PASS、訓練13,101,265全状態と過去評価開始の事前・事後分離PASS。CPU/CUDA CLI、人間先後自動合法入力完走。REPORT.md/summary.json/detail-audit.jsonへ結果・制約・再現CLI保存。探索を改善教師として採用する根拠なし、追加学習への利用自体は未検証。branch commit/pushを仕上げる。
