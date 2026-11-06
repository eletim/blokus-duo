# 4ch GPU更新改善と本学習・独立評価

main 6be347f基点。main/公開gen2/2048は変更しない。新モデルは4ch（両者占有＋両者接続可能点）、開始点専用平面なし。新規初期化だが既存heuristic priorを保持し、完全ランダムpolicyではない。

## 最適化の結論（確定）

native収集済みの基準に対し、同期集約＋rollout tensor常駐を採用。モデル、float32、Adam、minibatch16、loss係数、KLのstep単位早期停止、grad clip、更新量は不変。有限判定はoptimizer更新前に実施し、非有限なら更新しない。7指標を1回のD2Hへまとめ、Categoricalの重複引数検査を除去した。合法mask/有限性は別途検証する。元updateは参照として保持。

更新単体は2warmup/6交互反復、中央値baseline0.13620秒、同期集約0.13340秒、同期集約＋常駐0.12462秒（約1.09倍）。前回常駐単独では利益がなかったため、今回は同期集約との組合せを分けて測った。新規batchやprecision変更による見かけのiteration短縮ではない。

16iterationの収集＋更新pilotで、初回を除く中央値0.21910→0.21331秒（約1.03倍）。基準/nativeの前段階約2.1倍とは別の、小幅な追加高速化である。GPU更新自体を桁違いに速めたとは主張しない。

profileの46 optimizer stepsでは、stream同期1108→469、D2H506→46。H2Dは602→423。baselineのCPU kernel launch自己時間約37.5ms、device embedding backward約8.2ms、Adam device約10.2ms。profileには計測overheadがあり、別rolloutのprofile同士の時間比を速度比としない。定常速度は上記同量反復を用いる。メモリ・states/sec・steps/sec・保存込みwallは各JSON参照。

CUDA graphは決定的演算を有効にした現構成でbackwardのcaptureが失敗した（個別演算の原因特定は未完）。決定性を無効化して採用することはせず、不採用コードと失敗ログを保存。torch.compile/AMP/batch変更なし、compile初期費用なし。CUDA初期化/warmupは定常比較から分離。GPU使用率は成果指標にしていない。

小pilot16iterationで軌跡、model/past/optimizer、Python/NumPy/Torch/CUDA RNGが基準とbit一致。別process連続16対2＋追加14でも一致し、past交換4/8/12/16を跨いだ。workerはなく、保存境界は完了iteration。

## 本学習と評価

3seed各4096iteration/65536games、合計196608gamesを完走。生成＋更新＋保存wall2774.18秒（46.24分）、初期化・再開読込を含むrunner全wall2775.41秒。追加更新なし。4590668 learner states、554495 optimizer steps。CPU生成/pack400.46秒、収集推論＋同期660.09秒、update1604.25秒、履歴書込14.59秒（残りは制御・保存等）。学習条件は事前PLANどおり固定。3seedは初期化と自己対戦乱数が異なり、モデル/損失/報酬/pool/経験量を共有する。現在50%/過去25%/heuristic25%はゲーム割合で、learner state割合は自己対戦で両側を記録するため約66%/17%/17%。正確なseed別件数はtraining-audit.json。

計測・小pilotは各有限processのみ。主学習前のprofileから本学習開始まで約4分（実装時間も含む上界）、学習後のevent計測・全テスト約16秒と少数CLI確認で、30分の計算枠内。全ツール横断のGPU稼働秒は計測していないため精密な総GPU秒とは呼ばない。本学習90分枠も厳守した。

新validation32prefixで、criterion=.5対同seed学習前point+.25対heuristicpoint+.25対対応gen2pointを使用。候補1024/2048/3072/4096から4096/2048/4096を固定。gateは平均対学習前>.5かつ対heuristic>=.5。全seedを保持し、holdout結果で再選択していない。32prefixの選択にはノイズがあり、seed1751402は最新4096より2048が選択された。

fresh96prefix×両座席×3seed=576games/対照。CIは各prefix内の6ゲームをまとめて4000回bootstrapし、6ゲームを独立標本とは扱わない。すべてネットgreedy単体（同値は固定乱数）で探索なし。

|対照|point（勝1・分.5）/95% CI|平均得点差/95% CI|
|---|---|---|
|同アーキテクチャ学習前4ch|80.56% [77.26,83.77]|+7.09 [6.39,7.83]|
|固定heuristic|80.56% [77.26,83.77]|+7.10 [6.40,7.83]|
|公開gen2|48.00% [43.75,52.34]|−0.39 [−1.17,+0.41]|

学習前はheuristic priorにより固定heuristicとほぼ同じgreedy方策である。この2対照は独立な強さの証拠ではない。gen2は固定6ch公開モデル（対応1401/1402/1403、学習量49152/65536/65536games）。今回選択4chの経験量は65536/32768/65536gamesであり、全seedの配布モデルが同経験量という比較ではない。3seedとも最新65536gamesまでの学習は完了している。

|4ch seed|選択iteration|対gen2 勝/敗/分|point|平均得点差|
|---|---:|---:|---:|---:|
|1751401|4096|95/88/9|51.82%|+0.46|
|1751402|2048|79/102/11|44.01%|−1.34|
|1751403|4096|88/95/9|48.18%|−0.29|

seed別CIと全局結果はholdout-summary.jsonおよびholdout-*.json。4ch推論は約0.521ms/手、対照gen2は0.521–0.537ms/手（native観測、CUDA、合法手生成/転送/同期/出力を含む、モデル初期化除外）。思考時間制限を課した探索比較ではない。旧6ms/100ms教師との今回の比較はしておらず、一般的最強とは呼ばない。

## 分離・検証

全76 tests PASS。CPU/CUDAの各device内で、基準経路対最適化経路の同入力logit/value/loss/勾配/更新後重みをbit一致確認（CPU対CUDAのbit一致は主張しない）。step単位KL停止と非有限時の更新禁止も検証。CUDA別process再開16対2＋14（past4/8/12/16交換）PASS。選択3exportのCPU/CUDA出力は元checkpointとbit一致し、CPU専用環境/CUDA CLI対戦も完走・再生PASS。

過去156prefixファイル3046開始状態を新生成prefixから除外。全新ゲームを座標checkerで合法性検査し、終局得点/報酬/learner側を全件照合。action indexと列挙順の照合は256 learner statesごと。旧評価開始状態への訓練中の偶発的な再到達はseed1751402で2状態あり、これはprefix採用での重複ではない。評価bankはこれらも含む全訪問状態を除外して作った。

過去PPO訓練索引13101265状態をa011c78のSHAで参照し、新本学習・現在/前段階pilotと合わせて19677312状態を分離に使用。過去巨大履歴をコピーせず、和集合cacheは/tmpへ再生成する。新規/補助側のunique状態は6677060。validation/holdout開始状態の重複0、評価後の実訪問状態重複も45組4032gamesで0。算術・選択最大criterion・保存hash・推論呼出数/時間集計PASS。holdout prefix hashはa96315facbf09455f719730b625120f62400d179e283f034b628d9be808666cf。

## 判断と次案

**4chは学習前より明確に強くなったが、公開gen2を安定して上回る目標は未達。mainと推奨はgen2を維持する。** 高速化は同じ更新結果を保つため、それ自体を棋力改善の原因としない。4chと6chで初期化/データ/選択経験量が異なるため、gen2との差を4ch化だけの因果効果とも断定できない。

entropyは約2.3→1.0–1.2へ低下、value MSEは後半も約.63–.65。gen2向けvalidation曲線はseedにより非単調で、loss低下や無制限の更新追加を根拠に推奨変更できない。次はGPUのtiny minibatch/dispatch費用に対し、**minibatch64を別の学習条件として16と同ゲーム数・3seedで対照する有限pilot**を候補とする。optimizer stepsが変わることを明記し、実装等価の速度比とは呼ばない。gen2を主目的にした十分な新validationを先に固定し、速度だけでなく棋力低下がないかを判定する。改善は未検証であり今回実施していない。

高速化単独採用候補はcommit640bd3f（main6be347fに依存）。モデル候補はartifacts/ppo-four-gpu-modelsの3exportで、公開既定への置換は非推奨。main変更なし。

## 再現コマンド（最終結果は後述）

```bash
# 専用venv、任意native backendのbuild後。すべてrepo rootで実行。
python -m blokus.ppo_native build
PYTHONPATH=. python scripts/profile_ppo_update.py
PYTHONPATH=. python scripts/bench_ppo_gpu_final.py
PYTHONPATH=. python scripts/check_ppo_gpu_pilot.py
PYTHONPATH=. python scripts/prepare_ppo_gpu_exclusions.py
PYTHONPATH=. python scripts/run_ppo_four_gpu.py
python -m blokus.ppo_gpu_audit
python -m blokus.ppo_gpu_eval validation
# selection.jsonのgate=trueの場合だけ
python -m blokus.ppo_gpu_eval holdout
python -m blokus.ppo_gpu_eval audit
python -m pytest -q
```

既存出力を上書きしないため、再実行は新checkout/出力先で行う。単独seedの有限実行・再開は `python -m blokus.ppo_gpu_train --output PATH --seed 1751401 --until 512 --exclusions artifacts/ppo-four-gpu/excluded-starts.json`、続きは同引数に `--resume PATH/iteration-512.pt --until 1024`。untilは累積iteration。mode=fastは同期集約＋常駐、baselineは従来update。過去状態索引のgit objectはa011c78を参照するため、監査再現にはその履歴と前4ch pilotの69e4c7bが必要（shallow cloneだけでは不足）。公開推論自体は訓練履歴不要。


## 費用・速度補足

同量pilot（warmup除く各15iteration）は両経路とも682 optimizer steps。baseline→fastでgames/sec73.03→75.01、learner states/sec1592.75→1638.77、optimizer steps/sec201.57→207.39。update比較のpeak allocated GPU memoryは約118.5→169.3MB。速度上昇の代わりに約51MBを使用する。

CUDA event補足は1minibatch中央値forward0.462ms、loss0.288ms、backward1.393ms、norm0.031ms、optimizer0.092ms、collate/H2D/同期wall0.274ms、ログD2H wall0.032ms。padding/実候補数比3.68。event間同期を入れた診断なので、この足し算を定常throughputとみなさず、速度比には独立の交互反復を用いる。詳細はphase-events.json、update-profile.json、fast-profile.json、圧縮Chrome trace。

学習曲線はsummary.jsonのiteration/games/cumulative_wall/entropy/KL/value error/opponent WDLに保存。最適化計測は初期pilot分布、本学習は異なる進行分布であり、どの環境でも同じ速度比とは保証しない。
