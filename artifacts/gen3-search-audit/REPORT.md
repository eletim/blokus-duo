# 固定gen3 NN探索監査・改善比較

結論: **今回のB/Cは不採用。main/Web/Frozen gen3を維持。** 追加学習0。旧探索が単体より有利な小標本結果はあるが、人間より強いとはいえない。ユーザーの人間棋譜は取得できておらず、敗因を再現したとは主張しない。追加の設定探索はしていない。

## 本番監査
- Webの500/2000msはSearchAgentへ届く（コードとSpyテスト）。旧探索はroot/internal幅4、深さ2→4。2秒を指定しても4ply終了で停止する。
- negamaxは現在手番の終局WDL（勝+1、分0、負−1）。毎手turnは交代し、passを挟む同色の連続配置も符号が正しい。終局+15/+20を含む独立座標oracleと照合。
- 全rootの同深度終了時だけ選択を更新し、未完了反復は破棄。旧統計のfallbackには強制passが含まれる。本対戦の非pass着手にtimeout fallbackは0（pass-aware-stats.json）。
- PPO valueは学習時current/past/heuristic相手poolで実終局WDLを回帰したもの。得点差・minimax真値ではなく、相手/探索葉分布が変わる精度は保証されない。
- Webは棋譜をセッション内に保持し、ブラウザからdownloadする。リポジトリのbrowser-terminal-gameは自動試験であり、人間の今回の対局記録ではない。

## 固定条件と計画
PLAN.mdを評価前に保存。全3seed1751401/2/3、公開gen3 SHA不変、CPU/Python backend（本番と同じ）、torch threads1。B=root12/internal4の層バッチ・深さ2/4、C=B+残駒合計6以下の全合法手終局探索。いずれも500ms上限。Cの全探索以外は幅制限された近似探索である。深さ完成後のみ採用、act間の探索cacheなし。greedy手をrootから落とさない。

validationは新8prefix×先後×3seed、各比較48局、計240局。同じprefix集合を全対照へ使用。CIは同prefix内の両席/3seedをまとめたbootstrap（8クラスタなので小標本・探索的）。通常95%CIを以下に示す。確認試験の2比較補正CIもJSONに計算してあるが、validation5組の優越判定には流用しない。予定した独立確認32prefixは事前固定しただけで未使用。

|比較（左の成績）|point・通常95%CI|平均得点差・95%CI|実平均ms/手 左/右|
|---|---|---|---|
|old-policy|68.75% [54.17,83.33]|+2.979 [-0.854,+6.521]|169.6 / 0.8|
|B-old|52.08% [40.62,64.58]|-1.917 [-3.730,-0.145]|288.7 / 177.2|
|B-policy|55.21% [48.96,62.50]|+1.312 [-0.750,+3.146]|297.1 / 1.0|
|C-old|47.92% [35.42,61.46]|-2.562 [-4.417,-0.771]|288.5 / 178.2|
|C-policy|52.08% [46.88,59.38]|+0.896 [-1.083,+2.708]|296.9 / 1.0|

pointは勝1/引分0.5。seed別・席別成績はvalidation-summary.json。Bは対旧得点差が負で事前gate未達、Cも未達。相対順位で選ぶとBだが、採用候補を意味しない。旧対単体48局は32勝14敗2分。これは固定3seed・小標本の参考結果で、人間への優越や一般的最強の証拠でない。

## 費用と原因
- 比較対戦の実計測合計1112.7秒（約18.5分）。追加学習なし、60分枠内。対戦と監査/診断は分離。稼働Webは止めず、他の学習runは起動していない。
- 500msを超えた最大値は旧501.45ms、B503.90ms、C516.52ms。B対旧は23/652手超過、C対旧14/653手超過。p99は約500ms。硬い時間保証ではない。
- 旧の非pass1741手中1720手が4ply、Bは1129手中675手、Cは1125手中664手。幅を増やしたため深さ2へのfallbackが増えた。探索失敗でgreedyへ落ちたわけではない。
- Cのexact完了3件は全て強制pass。実際の選択がある着手での厳密解寄与は0。Blokusは残駒多数でも閉塞するので、残駒6以下の条件が実戦を捉えなかった。これを終盤厳密探索一般の無効性とはしない。
- 固定6ゲーム局面の500ms診断では、同じ幅4のbatch版は旧と6/6同じ選択で総時間比1.245倍（2000ms診断1.277倍）。これは小標本計算診断のみで、採用候補として追加対戦したものではない。
- root12版は500ms診断6局面のうち3局面で旧root幅外を選ぶが、元価値が高い手を真の最善とは呼ばない。同500msでは4/6局面が2ply止まり。2000msでは深さ4が完了し選択変更2/6、ただし2000msの棋力評価は未実施。
- Bの序盤pilotは観測生成約383ms、NN部分約101ms。profileはlegal_moves、候補features/packが主要費用。バッチNNだけの短縮では幅拡大の追加費用を吸収できなかった。

## 価値診断と限界
新しい独立6ゲームからroot6/探索葉24を取り、固定gen3 greedyで終局まで継続した実WDLと比較した。root MSE0.717・符号一致66.7%、leaf MSE0.628・62.5%。leafの段階別MSEは序盤0.971/中盤0.711/終盤0.201。6ゲーム内の葉は独立標本と扱わない。minimax真値の教師でも、人間対局の再現でもない。この小標本だけでは「探索葉への分布ずれが主因」と断定できない。

CPU batch logit差最大3.8147e-6が初期atol2e-6を超えた。評価前に例外をPLANへ記録してlogitのみ1e-5へ変更、valueは2e-6維持。bit等価を主張しない。固定木/選択一致、状態不変、部分反復破棄、独立小終盤WDLをテストした。

## 検証・保全
- 訓練union 27,533,566状態のSHAを照合。新開始局面は訓練/旧評価から分離。全240局を座標合法性checkerで再生、終局得点・集計一致。実訪問状態（終局含む）の訓練重複0。
- 初期の過去評価開始3558件の除外に加え、全ローカルblokus worktreeの194旧prefix bankも事後照合し交差0。過去巨大履歴は複製せずSHA参照。
- CPU-only全テスト98 passed / 4 skipped。最初は新worktreeにnative未buildだったため既存非NN系9テストが失敗、元ログを保持。既存手順でnativeをbuildして全体PASS。本NN対戦は全てPython backendなのでこのbuildで測定条件は変わらない。
- CPU人間CLIの先後両席を合法手入力で終局まで確認。FrozenモデルのSHA照合PASS。未採用なのでWeb/main向け統合候補は作らず、研究CLIのみ提供。

## 再現
```bash
# 必要なtorch/numpyを導入した環境で、リポジトリrootから
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/run_gen3_search_audit.py prepare
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 timeout 3500 python scripts/run_gen3_search_audit.py run
python scripts/run_gen3_search_audit.py audit
python scripts/diagnose_gen3_search.py
python -m blokus.ppo_native build
python -m pytest -q
python -m blokus.ppo_search_play --method B --budget-ms 500 --model-seed 1751401 --human-seat 0
python scripts/verify_frozen_gen3.py
```
prepare/auditは過去研究の訓練index `/tmp/ppo-lr64-training.npy` と既存bank参照を必要とする。index内容SHAはseparation-source.json、復元元はresearch/ppo-lr-64のtraining-auditと参照履歴。推論CLIはそのindexや研究履歴を不要とする。runは既存結果JSONがある比較を再実行しない。モデル変更や結果の後付け選択を防ぐため、再測定は別出力・別計画とする。

## 次の具体案
強さの主因は未特定。今回の根拠から次に優先するのは、**葉ではvalueに不要な全候補特徴を生成しない経路**を数値一致検証し、旧の完成済み4plyを先に確保してから同深度のroot拡幅へ余剰計算を使う対照である。現在は葉でも全候補policy入力を生成している。幅だけを広げて深さを落とす追加調整や長期再学習を直ちに繰り返さない。人間棋譜が得られた場合は別の診断資料として扱う。

cold-warm.jsonにはCPU専用の新processでimport約0.548秒、モデル構築/load約0.0058秒、初回/反復着手を分離保存した。OSファイルcacheは消しておらず、Bは旧の後なのでcoldの方式間速度比較には使わない。対戦ファイルのcold_load_secondsも新Agent構築時間であり依存import全体ではない。

固定診断root6局面はdiagnostic-roots.jsonへ保存。全30件の校正値/継続targetが同一になることを再生確認し、訓練indexとの交差0を確認した。診断費用を含め主計測60分枠内で終了。保存したconfirmation bankは未評価なので、その成績による調整もない。
