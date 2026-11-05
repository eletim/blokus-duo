# 終局PPO 2048：追加の学習AI

既存の低予算value/2plyと100ms教師は残し、デフォルトは変更しません。PPOは盤面・残駒・正確な配置セルから全合法手を直接評価するネット単体です。終局勝敗で自己対戦学習し、探索を付けていません。固定heuristic事前値も使います。3seedは同じ初期重みと学習条件を共有し、訓練乱数が異なります。

## 新しいcloneから実行

Python 3.12、PyTorch 2.7.1を使用。CPUだけでも推論・人間対戦できます。CUDA環境ではRTX5090で検証したCUDA12.8ビルドを利用できます。学習用のGPU/RNG/optimizerや研究branchの履歴sidecarは不要です。

```bash
git clone https://github.com/eletim/blokus-duo.git
cd blokus-duo
uv venv --python 3.12 .venv
# CPUだけの場合
uv pip install --python .venv/bin/python 'torch==2.7.1' --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python numpy pytest
# CUDAを使う場合は上のtorch導入を次で置き換える
# uv pip install --python .venv/bin/python 'torch==2.7.1' --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m blokus.ppo_play --model-seed 1401 --device cpu --human-seat 1
```

[PyTorch公式のバージョン別導入手順](https://pytorch.org/get-started/previous-versions/)にある2.7.1のCPU/CUDA12.8配布を使用しています。macOS/MPSは今回検証していません。`--device auto` はCUDAがあればCUDA、それ以外はCPUです。

```bash
# 学習AI対heuristic
.venv/bin/python -m blokus.ppo_play --model-seed 1402 --device auto
# 別seed同士
.venv/bin/python -m blokus.ppo_play --model-seed 1401 --opponent ppo --opponent-model-seed 1403
# 既存の6ms value/2ply、または100ms教師との対戦
.venv/bin/python -m blokus.ppo_play --opponent oldvalue --device cuda
.venv/bin/python -m blokus.ppo_play --opponent teacher100 --device cuda
# 人間が先手（0）、AIが後手（1）
.venv/bin/python -m blokus.ppo_play --model-seed 1403 --device cpu --human-seat 0
```

人間は表示された合法手番号を入力します。不正な番号は再入力、合法手がなければ自動pass。Ctrl-C/EOFで終了できます。例と既定値の1401は評価前に固定しており、今回holdoutで最良seedを選び直していません。CPUでも探索を実行できますが、同じms予算でGPUと同じ手・強さにはなりません。

推論checkpointは `artifacts/ppo-2048/seed-{1401,1402,1403}.pt`。`weights_only=True`で読み込み、学習履歴を参照しません。export元hashはmanifest.json。CPU/CUDAで元checkpointとの入力・logits/value・greedy選択bit一致を確認しました。

## 評価の読み方

旧実験のfresh96prefixでは3seed平均で旧6ms探索に得点差+2.104 [95%CI +1.272,+2.891]。今回別の96prefixで固定2048を再確認し、100ms教師とseed1401再開元512も比較しました。得点差CIは2座席と3seedを同prefix内で平均したcluster bootstrapで、更新seed母集団を代表するCIではありません。推論時間が違うため同時間比較でも、一般的最強の主張でもありません。

研究の詳細・訓練履歴は [research/ppo-learning-curve](https://github.com/eletim/blokus-duo/tree/research/ppo-learning-curve) および [research/ppo-4096](https://github.com/eletim/blokus-duo/tree/research/ppo-4096) を参照。mainには推論モデルと評価要約だけを追加します。

### 固定2048の別96prefixでの再確認

- 旧value6ms: point **56.08% [95%CI 51.82,60.16]**、得点差 **+0.696 [-0.195,+1.590]**。平均得点差の正の再確認は不確実。
- 固定100ms教師: point **44.97% [41.15,48.87]**、得点差 **−2.026 [−3.087,−0.988]**。教師の方が強い結果。
- 実平均: 旧6ms戦でPPO **1.218ms** / 旧探索 **4.690ms**。教師戦でPPO **1.346ms** / 教師 **70.388ms**。探索上限と実時間は異なる。
- seed1401の2048対再開元512: **147勝36敗9分、+7.536 [6.135,8.938]**。前回selected128との比較ではなく512への直接比較。

1344対局の合法性・算術・prefix単位CI・実訪問状態を監査し訓練重複0。旧6msへの改善幅には標本/seed差があり、100ms教師を上回っていない。2回の評価と今回の固定seedを保持し、良かったseedへの再選択をしない。全結果はevaluation-summary.json、試合再生と全履歴は研究branchに保存。

3seed×CPU/CUDA×26局面でexport前後の入力/logits/value/greedy選択bit一致。CPU/GPU AI対AI、人間先後の入力経路を終局まで検証した。推論ファイルは各約276KiB。
