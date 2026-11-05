# 固定PPO policy/valueの短時間探索: 採用見送り

公開gen2全3seedの重み（1401=3072、1402/1403=4096）を固定し、追加学習0で比較した。探索による強化は確認できなかった。main `a96d280`、既存ネット単体推奨、旧value/2ply、100ms教師は変更しない。

## 事前条件と方式

`PLAN.md`、`validation-lock.json`を参照。基点952d4cb、研究branch `research/ppo-short-search`。rootをpolicy上位4/8手、相手をpolicy上位2/3手に限定したbatched 2ply探索（6/25ms）。全rootの同じ深度が完了した場合だけ比較し、不完全・遅着バッチは捨てて同じpolicy greedyへ戻す。相手応手はroot視点のmin。PPO valueは手番側の終局勝敗期待値[-1,1]であり得点差ではない。終局は実スコア比較のsign、両者全置きも既知値。枝幅制限があるため完全minimaxではない。TTなし、試合ごと費用予測cold。

devで初期6ms費用予測が過大と判明し、validation前に実測準備費による予測へ変更した。初期結果と再測定はそれぞれ`dev.json`、`dev-cost-profile.json`に保全。`lock.json`のplan hashは開始時版、`validation-lock.json`は追記後の最終事前計画と実行ソースを固定する。

新validation1700000の32prefixを全3seed・先後交換で共有。候補設定はpoint→得点差→短予算で全seed共通選択。gateはpoint≥.52、得点差>0、各seed point CI上限≥.5、非強制手fallback<.5、監査PASS。初版未達後、事前登録した一度だけの修正としてvalue改善>0.15のときだけ元policyを上書きした。閾値・幅・checkpointを結果から再探索していない。両案ともgate未達のため96prefix holdoutと100ms教師比較へは進まなかった。これは小標本validationであり確認的holdoutの成績ではない。

## 同checkpointネット単体との結果

各行192局=32prefix×先後×3seed。point=(勝ち+引分/2)/局数。CIは同prefixの先後と3seedを内側で平均したprefix単位bootstrap95%。seedや個々の対局を独立prefixとして数えない。

|方式|point [95%CI]|平均得点差 [95%CI]|探索実ms/手|非強制手fallback|上限超過率|
|---|---|---|---:|---:|---:|
|初版6ms|.500 [.471,.531]|+0.547 [-0.047,+1.120]|2.160|61.38%|0%|
|初版25ms|.435 [.378,.490]|−2.484 [-3.792,−1.224]|7.987|3.84%|0.034%|
|margin .15、6ms|.513 [.497,.531]|+0.245 [-0.063,+0.562]|2.143|61.74%|0%|
|margin .15、25ms|.458 [.393,.526]|−0.906 [-2.115,+0.281]|8.008|3.47%|0.034%|

6msの小さな差はCIが改善なしを含み、fallback基準も未達。25msでは多くの探索が完了しても元policyより弱くなった。平均実消費は上限より短い。ネット単体の実時間はJSON参照（約1.3ms）であり、同時間対戦と呼ばない。

|seed|初版6ms point/差|初版25ms point/差|margin6ms point/差|margin25ms point/差|
|---|---|---|---|---|
|1401|.492/+0.234|.375/−3.688|.523/+0.312|.508/−0.859|
|1402|.453/−0.125|.438/−2.250|.492/−0.406|.359/−2.391|
|1403|.555/+1.531|.492/−1.516|.523/+0.828|.508/+0.531|

seed別CI・W/L/D・全手実時間は`validation-*.json`と`summary.json`。最終選択（採用ではない）はmargin6ms。補助の旧6ms value対戦はpoint .643 [.589,.701]、差+3.370 [+2.026,+4.839]。これは同ネット単体より強くなった証拠ではなく、元gen2の強さを多く保持した比較。補助192局を合わせvalidation合計960局。

## 計算費用とvalue診断

追加学習・更新・checkpoint生成は0。960対局の逐次計測wall合計96.35秒。bank作成、dev36ゲーム105root、validation、最初の全対局監査を含むrun区間は約220秒（lock.started→complete.finished）。その後の全テスト、追加分離監査、CLI確認、別microprofile、文書化はこの区間外。過去履歴はSHA参照し再複製していない。

25ms初版の全手平均はroot処理1.345ms、子局面合法手/特徴生成5.464ms、相手policy batch0.509ms、葉value batch0.553ms。平均完了root6.47/応手18.42、完了2452/2959呼出。元policyの着手を62.15%で置換、margin後は26.17%。CPU候補生成が主要費用であり、GPU演算を増やすだけでは解消しない。

別microprofile（devの10局面、対戦と分離）: root観測生成平均1.165ms、最大8子局面の観測生成6.421ms、batch転送/同期0.127ms、GPU forward kernel単体0.637ms。CUDA peak allocated最大39,215,104bytes（モデル・入力等を含む、ドライバ予約全体ではない）。対戦の時間には合法手生成・転送・同期・推論・fallbackを全て含む。OS等のhard deadline保証はない。初版/修正25msで各1手だけ超過した。

dev1690000は36独立ゲーム、各最大3root。相手self/heuristic/oldvalue各12ゲームのvalue MSEは.728/.705/.703、ゼロ予測基準は1.000/.917/1.000。ゲーム単位95%CIはそれぞれ[.428,1.088]/[.276,1.196]/[.245,1.326]。全体の勝敗予測がゼロより良い傾向でも、同局面候補間の順位や探索分布での校正は保証されない。特にopponent poolから変わった相手の条件付き小標本であり、真のゲーム価値の推定精度とは断定しない。

## 検証

- 全120テストPASS（新規8）。同一value head、手番反転、pass、終了15/20ボーナス、合法候補、相手min backup、同値/未完了/遅着fallback、状態不変、cold reset。
- 独立座標合法性checker＋全探索で小終盤8条件の値/最善手集合一致。全配置判断が2ply以内に終わる人工小inventoryに限定し、中盤の厳密性は主張しない。
- 公開3seed checkpoint SHA固定。dev105rootのCPU別経路valueとCUDA保存値は誤差2e−6以内。
- 960対局すべて合法再生、終局スコア、先後ペア、得点符号、prefix CI再計算、呼出数/時間集計PASS。
- 全訓練13,101,265訪問状態をindexチェーンSHA検証。dev/validation開始状態の過去評価開始との事前分離、実訪問状態の訓練/過去評価開始との事後分離PASS。初期空盤面からのCLI煙テストは評価bankに含めない。
- CPU/CUDA AI対AI完走、人間先後を合法入力で自動駆動し完走/再生確認。`cli-audit.json`。固定重み推論のみのためoptimizer/RNG学習再開の追加検証対象はない。

監査ファイル: `evaluation-audit.json`、`detail-audit.json`、`tests.log`。この工房側再生監査を、秘書の独立実装監査とは呼ばない。

## 判断と次案

探索の改善は未確認、main/gen2推奨維持。限定応手幅による見逃し、学習時poolの状態価値と探索葉の分布差、候補間のvalue誤差をmax/min選択が増幅する可能性は残る。本比較だけで原因の寄与は分離できない。単体policyの良い手をvalue選択が失わせ得ることは対照で確認した。

本探索を新しい自己対戦の改善教師として採用できる根拠はない。教師として使った再学習そのものも未検証。次へ進むなら幅/閾値の追加微調整より、独立した学習用ゲームでpolicyが実際に選ぶ候補と探索葉の終局結果を収集し、状態valueの候補間識別と校正を先に診断することを推奨する。今回のvalidation/過去holdoutを教師データへ転用しない。追加の有限自己対戦学習を行うかは秘書判断に返す。

## 再現

リポジトリrootで専用venvを使用:

```bash
.venv/bin/python -m pytest -q
PYTHONPATH=. .venv/bin/python scripts/audit_ppo_short.py
PYTHONPATH=. .venv/bin/python scripts/profile_ppo_short.py
.venv/bin/python -m blokus.ppo_short_play --model artifacts/ppo-gen2/seed-1401.pt --device cuda --budget-ms 25 --margin 0.15 --opponent self
.venv/bin/python -m blokus.ppo_short_play --device cpu --budget-ms 6 --human-seat 0
.venv/bin/python -m blokus.ppo_short_experiment run
```

`run`は保存済み対局を再利用する有限run。履歴indexは研究branchの既存成果を参照し、推論CLIはそれらを要求しない。新規測定のときは既存成果を保全した別worktreeで新artifact root/seedを事前固定する。deadline方式はOS/CPU/GPU負荷で着手が変わり得るため、固定seedだけで時限制御対局のbit一致は保証しない。固定モデル、prefix、各着手/値、設定と全時間を保存して検査可能にした。公開モデルのseedを替える場合はmodelパスの1401を1402/1403へ変更。研究用CLIでありグローバル既定AIは変えていない。
