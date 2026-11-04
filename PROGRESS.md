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
