# 公開gen3固定版 — 2026-11-06

固定タグ: `gen3-frozen-2026-11-06`（annotated tag）。コードと3モデルを同じcommitへ固定する。タグの参照先commitは `git rev-parse gen3-frozen-2026-11-06^{commit}` で取得できる。タグは管理者が移動可能なので絶対不変の保証ではない。取得したcommit IDと [SHA256 manifest](artifacts/ppo-gen3/frozen-manifest.json) を固定参照として保管する。

**gen3はニューラルネットのPPO**（heuristic priorあり）。非NNの学習済み線形評価Lとは別方式。既定seedは1751401、他は1751402/1751403。公開重み・既定・Web動作に変更なし。4×14×14の盤面入力（両者占有と接続可能点）に残駒等49特徴、候補39特徴・配置セル・合法maskを組み合わせる。開始点専用チャネルはなく、初手の接続可能点は(4,4)/(9,9)。CLIのネット単体には探索なし。Webの探索モードは追加処理である。

採用元はresearch/ppo-lr-64 commit67482585のB、minibatch64/LR3e-4、5120iteration・各81920games。確認commit16e3baa。3ファイルは各281805 bytesの `terminal-ppo-inference-v2` 推論exportで、既存manifestとbyte一致。ファイル内の歴史的 `experimental: true` metadataも変更していない。optimizer、過去方策、訓練RNGがないため、このexportは学習の厳密再開用checkpointではない。研究履歴・sidecar・nativeビルドは推論に不要。

## CPUで復元

検証環境: Linux x86_64、Python3.12.14、torch2.7.1+cpu、NumPy2.5.4。全依存の固定版は `artifacts/ppo-gen3/frozen-cpu-environment.txt`。Python3.12を用意して実行する（CUDA不要）。

```bash
git clone --branch gen3-frozen-2026-11-06 --single-branch https://github.com/eletim/blokus-duo.git blokus-gen3-frozen
cd blokus-gen3-frozen
git rev-parse HEAD
git rev-parse gen3-frozen-2026-11-06^{commit}
python3.12 -m venv .venv
.venv/bin/python -m pip install 'torch==2.7.1+cpu' --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r artifacts/ppo-gen3/frozen-cpu-environment.txt
.venv/bin/python scripts/verify_frozen_gen3.py
# CPU / Python backend、AI対heuristicを終局まで
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m blokus.ppo_play --generation gen3 --model-seed 1751401 --device cpu --backend python
# 人間先手（後手は --human-seat 1）
.venv/bin/python -m blokus.ppo_play --generation gen3 --model-seed 1751401 --device cpu --backend python --human-seat 0
```

依存配布サービスが将来終了する可能性まで保証するものではない。CUDA環境では既存検証済みtorch2.7.1+cu128も使用可能だが上記CPU lockとは別環境。異なるハード・依存環境での浮動小数bit一致は保証しない。

3モデルはGit実体としてcloneに含まれる（LFS pointerや外部ダウンロードではない）。manifestにサイズ・SHA256・Git blob ID・入力仕様・由来を記録。軽い照合はPython標準ライブラリのみで実行できる。新cloneで全3seedをCPUロードし、明示gen3対戦の合法性と終局得点を再生検証した記録は `artifacts/ppo-gen3/frozen-validation.json`。

## 保全運用と限界

今後gen3重みは上書きしない。変更する学習モデルには新世代名・新manifest・新タグを使い、このタグも上書きしない。固定commitから復元すれば将来mainの既定が変わっても同じコードと重みを取得できる。

既存の固定256prefix・3seed確認ではgen2にpoint54.13%（補正CI51.37–56.87%）、平均得点差+1.20。これは固定評価分布での結果で、一般的最強や全seedへの一般化を意味しない。今回のFrozen化は追加学習・モデル選別・棋力改善ではない。
