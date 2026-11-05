# CPU候補処理の等価高速化

実用的な速度改善を確認。合法手生成・ルール・特徴値・乱数消費・モデルは変更せず、最大2系統の重複計算を減らした。main `a96d280`は未変更。main向け候補は別branch **integration/ppo-cpu-throughput / c901d09**（a96d280から直接分岐）として準備し、統合判断を秘書へ返す。

## 対象と設計

研究基点bd962d7、branch `research/ppo-cpu-throughput`。初期profileで合法手約.212ms、12特徴.226ms、39特徴pack .397ms（12特徴含む）、observe .707ms、盤面encode .019ms。16ゲームbatchのcProfileでも候補特徴/packが支配的だったため、以下の2系統だけ変更した。

1. `features`: 配置固有の中心距離/開始点距離を再利用。局面内で一定の駒数・盤面面積差・frontier個数をloop外へ移動。
2. `inputs/observe`: 同じ配置の27追加特徴と5セルindexを再利用。

`placement_geometry.geometry(Move)`は不変tupleのみを返す最大32768件LRU。状態をkeyに使わず、board/remaining/turn/last/passesをcacheしない。NumPy出力は毎回新規所有しviewを共有しない。全13,729配置を保持して約11.1MB（tracemalloc、10.6MiB）。native依存なし、従来どおりPython/NumPyのCPU-only利用が可能。

`engine.py`は基点とbyte一致。列挙集合/順序/遷移APIを変えていない。旧`features/inputs/observe/legal_moves`を`cpu_reference.py`に保存し、元commitのASTと一致することを監査した。

## 性能

新seed1730000〜1730011の12ゲームから序中終盤・passを保存し、同一125保存時点（空盤面等の重複を含む）・同じgen2 seed1401重みで比較。各operationは1bank warmup後、3巡を計測し、旧新順序を交替して4反復。モデル初期化/ロード/warmupは除外。表は反復の中央値。

|処理|旧ms/局面|新ms/局面|速度比|
|---|---:|---:|---:|
|合法手列挙|.213|.213|1.00×（未変更）|
|12特徴|.224|.101|2.23×|
|39特徴pack（12特徴含む）|.397|.184|2.16×|
|盤面encode|.019|.019|1.00×|
|observe全体（合法手含む）|.708|.451|1.57×|
|ネット単体 CPU|.925|.666|1.39×|
|ネット単体 CUDA（同期/転送含む）|1.200|.929|1.29×|

新clone・独立CPU-only torch2.7.1+cpuでも、単体推論中央値 **.920→.654ms（1.41×）**。これは同一環境内の旧新比較。CUDA版とCPU版を同条件の棋力比較として扱わない。

別測定でcacheを空にしたbank初回は.514ms/局面、warmは.441ms、旧版は.706ms。初期化は無料ではなく、cold時は配置幾何を初めて計算する。メモリ測定は速度測定と分離した。

### 16games batchの収集＋更新

各組は同じgen2モデル、pastコピー、fresh Adam(.0003)、seedから開始し、**2iterations×16games**。3組で旧新順序を交替。全モデル初期化と1batch収集＋更新warmupを除外。wallには共通の観測hash検査も含む。各段階の中央値を別に示すため合計とwallは厳密には一致しない。

|32ゲームあたり|旧|新|
|---|---:|---:|
|収集＋更新wall|1.305秒|1.024秒|
|収集のみwall|1.035秒|.777秒|
|収集games/sec|30.93|41.19|
|CPU合法手/特徴/pack wall|.897秒|.639秒|
|GPU推論・同期等wall|.119秒|.118秒|
|更新wall|.255秒|.239秒|

収集速度約1.33×、収集＋更新約1.27×。GPU欄はcollate/転送/同期を含む呼出wallであり、純GPU kernel時間ではない。GPU側を変更したとは主張しない。短い3反復の実測であり、ハードウェア・局面・batchによって比率は変わる。

最初の測定は更新kernelのwarmup不足を避けるため再測定した。初回結果は`training-initial.json`に保全、最終値は`training.json`。実装やseedは変更しない。合計は計測192ゲーム×2回＋warmup16×2回=**416ゲーム**の有限等価性/性能検証。追加本学習・配備checkpoint生成は0。過去の巨大履歴を複製していない。

## 等価性・検証

- 全21駒・全向き・盤端を含む**13,729全配置×両手番**で12特徴/39特徴/セルmask/観測配列のdtype・shape・全byte一致。
- 新12ゲームの125保存時点で旧列挙順一致、独立座標checkerで合法性確認。pass・終局拒否・遷移・状態不変・配列を書き換えても次の観測が壊れないことを検査。
- CPU/CUDAそれぞれ旧新のlogits/valueと着手がbit一致。CPU対CUDA相互のbit一致を主張するものではない。
- 短いPPO全3組で全軌跡、全観測入力、model/past/optimizer、Python/NumPy/Torch/CUDA RNGがbit一致。更新ログも秒数以外完全一致。報酬・action index・合法再生・終局スコア監査PASS。
- 研究branch全**126 tests PASS**。
- main候補のCUDA環境全**64 tests PASS**。
- main候補の新規clone・CPU-only環境全**63 PASS / CUDA専用1 skip**。AI対AI、人間先手/後手（合法番号を自動入力）のCLI完走と再生確認。`fresh-cpu-human.json`にCPU-only/clone import元を保存。
- oracle出自、engine不変、保存ゲーム、ベンチ算術を`audit-summary.json`に保存。工房側検証であり秘書による独立実装監査とは呼ばない。

## main用最小差分

`integration/ppo-cpu-throughput`は研究branch全体をmergeせず、現mainから以下だけを変更した。

- production: `blokus/ai.py`、新`placement_geometry.py`、公開推論専用`ppo_policy.py`。
- 比較oracle、5等価性テスト、保存12ゲーム、履歴不要ベンチ、README。

mainは研究用`terminal_ppo.py/root_policy.py`を持たないため、公開推論側へ同じ変更を移植。geometry/aiは研究版とSHA一致し、推論側は旧新出力一致を独立に確認。モデル・世代既定・グローバルAI既定・CLI仕様は変更しない。候補commit **c901d09**を通常cherry-pick/merge可能。既存main自体へのpushは行わない。

## 制約と判断

棋力改善ではなく等価な固定計算の高速化。**時間制御探索は同じ上限内の展開数や打切り位置が変わり、着手も変わり得る**ため同挙動保証対象外。今回は探索成績を再選択していない。

新observeでは合法手生成の割合が増えたが、今回さらに3箇所目へ手を広げない。共有cacheは不変Moveだけに限定し、状態key不足の問題を避けた。実用的な推論/生成改善があるため、この最小差分のmain採用を提案する。main統合と次の学習予算判断は秘書へ返す。

## 再現コマンド

研究branch、リポジトリroot:

```bash
.venv/bin/python -m pytest -q
PYTHONPATH=. .venv/bin/python scripts/bench_cpu_throughput.py micro --label rerun
PYTHONPATH=. .venv/bin/python scripts/bench_cpu_throughput.py training
PYTHONPATH=. .venv/bin/python scripts/profile_geometry_cache.py
PYTHONPATH=. .venv/bin/python scripts/audit_cpu_throughput.py
```

`training`は32ゲーム×旧新×3組＋16ゲームwarmupの有限run。再実行時は既存結果JSONを先に別名保存する。

main候補branchを新規cloneした場合（訓練履歴不要）:

```bash
python -m pytest -q
PYTHONPATH=. python scripts/bench_ppo_cpu.py --device cpu
python -m blokus.ppo_play --device cpu --opponent ppo
python -m blokus.ppo_play --device cpu --human-seat 0
python -m blokus.ppo_play --device cpu --human-seat 1
```

CPU専用PyTorchで動作確認済み。CUDA環境では`--device cuda`も可。今回システム既定Python3.14にはensurepipがなかったため、既存環境と同じPython3.12で新venvを作成した。システムPython/村本体は変更していない。
