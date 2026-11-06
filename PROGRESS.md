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

## 100ms教師の限定統合
main eb96a90とd3c4698の依存差分を確認。engine/ai/research/value_searchと固定checkpointは同一。deep_search/refinement_searchを原本のまま追加し、独立終盤oracleを含む関連テストを追加（テストのBASE importのみ定数化）。研究用生成/学習/評価モジュールと巨大生データは取り込まず、専用teacher_play CLIと固定設定/2標本要約を追加。53 tests PASS、CUDA CLI対局完走 scores -22/-29、教師超過0。低予算デフォルト変更なし。生結果は保存済み研究branchを参照。

PPO2048推論追加: 3seedの固定モデルを履歴不要へexport、CPU/CUDA出力bit一致。新96prefix再確認/1344対局を監査。新clone CPU専用2.7.1+cpu/人間先後/各seed/既存探索対戦とCUDAを確認、main57tests PASS。既定例1401固定、既存デフォルト維持。研究履歴はresearch/ppo-4096参照。

## 新世代PPO提供
- 固定3072/4096/4096を凍結、新96prefix×3seed×3対照×先後=1728局を単独計測。対2048 +3.9583 [3.2031,4.6650]、旧6ms +2.9184 [2.0955,3.7049]。教師100ms +0.7674 [-0.2366,1.7484]、優位未確定。実訪問訓練重複0、合法性/算術/SHA/過去prefix分離PASS。
- 公開コードは80a5179を通常cherry-pickした22801aa。PPO CLI世代gen2既定、seed1401維持、旧2048は--generation 2048。旧グローバルAIデフォルトと100ms教師維持。学習履歴なし。
- 3seed×CPU/CUDA×26局面export bit一致。新cloneでCPU専用torch2.7.1+cpu、6推論テストPASS、人間先後・AI・世代対戦/旧探索をCPU/CUDA各13局再生PASS、main全59tests PASS。今回は追加学習なし。

main統合: c901d09→8c40e42をfast-forward。コードは候補と同一、既存72テスト/CPU clone67 PASSを引用。main上4ch/native CUDA学習CLIを1iteration16gamesだけsmoke完了。gen2 CUDA/2048 CPU対戦完走。READMEへnative build必須・Python fallback・CPU生成由来の短期測定速度比を明記。公開モデル/既定維持。本学習・重い再ベンチなし。

## gen3公開統合
最新origin/main6be347fからintegration/ppo-gen3作成。研究全体はmergeせず、固定B/5120/1751401-3のinference exportを確認16e3baa models.jsonとSHA完全一致で追加。gen3既定、旧世代/seed別名互換、global AI default維持。640bd3f由来dense同期集約update＋minibatch指定のみ抽出、raggedなし。LR/batch再開継承、明示変更履歴を実装。79tests PASS、CPU/CUDA 4vs2+2別process再開bit一致（past交換含む）、設定変更拒否/履歴確認。新clone公開CLI検証へ進む。
新clone: CPU-only 76passed/4CUDA skipped、CUDA80passed。gen3/gen2/2048のCPU/CUDA AI対戦・人間先後18経路PASS。旧metadataなしcheckpoint互換追加試験でPath型resume config JSON問題を修正。公開gen3は同じSHA、旧globaldefault差分なし。INTEGRATION.mdに依存/検証/再現/時間の読み方を保存。通常FF統合へ。

2026-11-07 NN Web探索監査開始。Frozen gen3/main保持、research/gen3-search-audit。設定伝達とWDL符号をコード確認、5局面費用pilot保存。PLANに最大2候補・事前gate・有限標本を固定。追加学習なし。人間棋譜は未取得。
NN探索: 訓練index 27,533,566状態（SHA cdd1f0f1...）と過去評価開始3558件から新bank分離。validation開始。新候補はroot12/internal4、深さ2/4同完成条件。CPU batch logit差3.8147e-6を記録しatol例外をPLANへ明示（棋力gate不変）。8件の対象テストPASS。追加学習0。
2026-11-07 NN探索有限比較完了。validation240局、対戦実wall1112.7秒。B対旧point52.08%/得点差−1.917、C対旧47.92%/−2.562で事前gate未達。独立確認・main更新なし。全棋譜/得点/終局含む訓練分離監査PASS（重複0）、CPU-only98testsPASS/4skip。CLI人間先後完走。診断root再生一致、Frozen SHA不変。詳細artifacts/gen3-search-audit/REPORT.md。追加学習0、計測run終了。
2026-11-07 新規cycle research/gen3-value-leaf（基点99cb70c）。value専用入力/forwardを実装、1188値比較で差0、102完成木着手一致。交互費用pilot192計測でPython同完成4ply A320.3ms/E97.9ms。Dは4ply保持後8→12拡幅に固定。新16prefixの500ms validationと新8prefixの2000ms補助、gate通過時のみ新32prefix確認をPLANへ事前固定。追加学習0、Frozen/main/Web保持。
2026-11-07 value-leaf cycle完了。追加学習0、528対局wall1949.98秒。Eの同完成木1188値差0/102着手一致、Python交互pilot320.3→97.9ms（3.27x）、native92.1→39.6ms（別要因）。Dは500ms D-A point47.92%/差-0.104、D-policy56.77%/差+0.8125、旧探索への事前gate未達。拡幅>=8約91%、非pass全4ply完成でも改善未確認。2000ms補助もD-A45.83%/差-1.75。新32prefix確認は未使用、条件救済なし。
研究CPU全104PASS/4skip、全528棋譜座標合法性/pass/得点/算術/CI/hash監査PASS、評価訪問状態の訓練重複0。E最小候補integration/gen3-value-leaf-E c9dddd5c97f1a27c5deae13b4e4e3a28e1525df7（main4f13460基点）、新cloneCPU全96PASS/4skip（既存native依存のためbuild後）、ブラウザ人間両席PASS。DのWeb候補なし。main/Frozen/稼働Webは未変更。REPORT/verification.jsonに費用・限界・再現コマンドを保存。
2026-11-07 E integration separated: main8fb3e9d, rollback4f13460, live policy/search/linear both seats smoke PASS, Frozen unchanged. New research/gen3-deeper-value: fixed width4, depth2→4→6, completed-stage fallback, conservative TT bounds/exact caching, no learning. Bound-window review corrected before validation; both pilot versions preserved, fixed-tree4/6 and terminal/pass/timeout tests4PASS. New6-game diagnostic found one estimated widening gain with no WDL gain under either greedy or fixedE2 continuations (not minimax, correlated). Main500ms selected from cost only,16prefix validation2pairs/3seeds, gatepass only32prefix confirmation. PLAN frozen.
2026-11-07 fixed-width deeper cycle complete. E public code integration8fb3e9d + published-doc2adc36f, service active and policy/search/linear both seats smoke PASS, Frozen gen3 unchanged. Research H width4/depth2→4→6/500ms, no training. Validation192games passed predeclared gate; fresh32prefix confirmation384games: H/E point62.7604%, Bonferroni97.5%CI57.2917–68.2292, diff+2.4167; H/policy68.75%, corrected61.1979–75.7813,diff+4.6875. H/E mean262.69/68.92ms (same ceiling, not same consumed time), H6ply1850/2487 nonpass, otherwise complete4. Max550.09ms (aux551.54), no hard deadline claim.
6-source-game diagnostic45candidates/180continuations: only1 added-root estimate gain, no continuationWDL gain, not minimax truth/independent180samples. Full576game+180diagnostic coordinate/score/hash/aggregate replay PASS, training overlap0, phase visited overlap0. Research111PASS/4skip; minimal candidate0740f31c19e03cefdc8a7747c573eca9dcf14216, CPUclone100PASS/4skip, synthetic humanCLI both seats complete. H not applied to main/Web. Match wall2416.86sec/40.28min, finite run stopped. REPORT/reproduction/verification saved.
