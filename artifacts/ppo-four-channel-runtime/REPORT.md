# PPO 4チャネルと生成基盤高速化

新学習の盤面入力を自分/相手の占有・自分/相手の接続可能点の4チャネルにした。空盤面の接続可能点は(4,4)/(9,9)で、エンジンと実入力を照合した。開始点専用2平面は入力しない。49次元状態・39次元候補特徴、ルール、報酬は維持。公開gen2/2048、seed1401、既存CLIの既定値は変更していない。

## 同量処理の比較

RTX5090、torch2.7.1+cu128、float32、16games収集＋同じPPO更新。既存の配置cache高速化済み実装を参照とした。各条件3反復、順序を交替。各経路の1収集＋更新をwarmupとして除外し、初期化と保存も別計測。各チャネル内では軌跡・入力・乱数・更新数・重み・optimizerがbit一致した。4chと6chの学習軌跡は異なるので、そのwall差をチャネル削減の効果としない。

|入力/経路|収集秒|更新秒|合計秒/16games|games/s|同入力参照比|
|---|---:|---:|---:|---:|---:|
|6ch Python参照|0.364|0.148|0.512|31.26|1.00|
|6ch native|0.093|0.145|0.238|67.36|2.15|
|4ch Python参照|0.327|0.130|0.458|34.97|1.00|
|4ch native|0.087|0.131|0.218|73.38|2.10|

各列は反復の中央値なので厳密な加算一致はしない。6chのlearner states中央値379、optimizer steps中央値45、候補総数62,456、最大候補数730。learner states/sは762→1,642、optimizer steps/sは87.9→184.8。更新数は既存KL早期終了を含め経路間で一致。全行・各反復はbenchmark.json、集約はsummary.json。

支配していたPythonの全配置走査・候補特徴生成をC11の1回の走査へ融合。合法順序を維持し、各出力は新しい配列へコピーする。状態cacheは使わず、不変配置tableだけを共有する。6ch収集のCPU生成/packは295→32ms。nativeの内訳は合法列挙＋特徴16.7ms、観測pack12.5ms、転送/pack/同期8.3ms、推論＋出力同期45.1ms。融合内の合法列挙と特徴は壁時計で完全分離できないため、phase-profile.jsonに同じ局面での単独列挙と融合呼出の補助測定を保存した。更新約145msが新しい主な費用である。

rollout全体をGPUへ一括packし、minibatchを再利用する経路も実装してbit一致を確認したが、6ch合計0.238秒と差がなく、peak allocated memoryが約120→174MBに増加したため既定には採用しなかった。--packedで再測定可能。batch数/精度/損失/更新順序は変更していない。CPU並列worker、torch.compile、非同期policyは導入していない。

同一保存batch（16状態、最大候補553/794等）で4/6chを交互に測定したGPUモデルforward中央値は0.469/0.471ms、backwardは1.047/1.049ms。10回warmup後のCUDA event測定で、転送とloss構築を除く。4ch化自体の有意な高速化は示していない。全推論は51保存局面で4反復、合法生成/転送/同期込み。6ch CPUは約0.715→0.250ms、CUDAは約0.992→0.519ms。同一checkpointの着手は一致。性能値はこのCPU・局面分布に限定される。

保存は約2.3ms。native buildは1.40秒（system compilerがなく専用venvへziglang0.13.0を導入）。fresh processではimport0.629秒、不変table0.106秒、CUDA/model/optimizer初期化0.632秒、初回iteration0.550秒、process開始から初回終了1.918秒。これら初期費用を定常速度へ混ぜていない。生成/学習/計測は同時実行していない。

## 互換性・検証

- Python経路と旧collect/update参照を保持。nativeは任意build、autoは未build時にPythonへ戻る。公開推論のbackend既定は従来のpython。
- 旧terminal-ppo-inference-v1は6ch限定、新v2はchannelsを明示してconv形状と照合。6ch重みを黙って4chへ変換しない。新runtime学習checkpointも独自format。旧研究terminal_ppo/ppo_scaleの6ch学習再開経路は残す。
- 134 tests PASS。全21駒/初手414手/盤端/pass/終局/座標checker/配列所有権/並列呼出のbuffer分離を確認。
- 6/4chとも参照/native/residentで3反復の軌跡・入力・更新後model/optimizer/RNGがbit一致。CPU/CUDA推論の同backend比較も一致。
- CUDA別processで連続4iterationと2＋保存再開2iterationがmodel/past/optimizer/Python/NumPy/Torch/CUDA RNGまでbit一致。past交換を跨ぐ。CPUは連続2対1＋再開1が一致。workerは存在せずcheckpointにNoneを明記。保存境界は完了iterationのみ。
- 4ch有限pilotは64iterations/1,024games/2,928 optimizer steps、14.54秒。全対局再生、合法性、終局報酬、learner側だけの更新、export一致を監査PASS。勾配有限、重み変化と初期局面policy KL=0.000960を確認。棋力は測っておらず推奨モデルではない。

GPU予算60分に対しpilot14.54秒、主benchmark測定18iterations＋warmup6iterations、kernel/inference profileと短い再開・テストのみを実行。追加本学習はない。各測定wallはJSONに保存。全ツール実行を一括計時していないため厳密な合計GPU稼働秒は主張しない（全短期測定は予算内）。

## 再現

リポジトリrootで実行する。初期化費用を含めたい場合はcold-start scriptを別processで実行。

```bash
# C compilerがなければ専用venvにziglang==0.13.0を入れる（任意）
.venv/bin/python -m blokus.ppo_native build
PYTHONPATH=. .venv/bin/python scripts/bench_ppo_runtime.py benchmark
PYTHONPATH=. .venv/bin/python scripts/profile_ppo_runtime.py
# full inference profileは先にbenchmarkと下記pilotを実行し、
# artifacts/ppo-four-channel-runtime/pilot/inference.ptへ実験用exportを用意する
PYTHONPATH=. .venv/bin/python scripts/profile_ppo_inference.py
PYTHONPATH=. .venv/bin/python scripts/profile_ppo_cold_start.py
.venv/bin/python -m pytest -q
.venv/bin/python -m blokus.ppo_runtime --output artifacts/my-four-channel-pilot --channels 4 --backend native --device cuda --iterations 64 --cap 600
.venv/bin/python -m blokus.ppo_runtime --output artifacts/my-four-channel-pilot --resume artifacts/my-four-channel-pilot/latest.pt --channels 4 --backend native --device cuda --iterations 68 --cap 60
.venv/bin/python -m blokus.ppo_play --checkpoint artifacts/my-four-channel-pilot/inference.pt --backend auto --device cpu --human-seat 0
# 公開済みモデルは従来どおり6ch
.venv/bin/python -m blokus.ppo_play --generation gen2 --model-seed 1401 --device cpu --backend auto
```

native build前でも--backend python/autoと--device cpuは利用可能。CPU-only環境はCPU版PyTorchを入れる。main統合候補には本実装・参照・テスト・再現bench・本要約だけを含め、pilot訓練履歴は研究branchを参照する。処理高速化であり棋力改善ではない。時間制御探索では探索量が変わるため着手同一を保証しない。
