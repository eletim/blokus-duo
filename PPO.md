# 終局PPO：世代を選べる学習AI

既存の低予算value/2plyと100ms教師は残し、既存コマンドのデフォルトは変更しません。現在のPPO専用CLI既定は **gen3 / seed1751401**。旧世代は `--generation gen2` / `--generation 2048` で選べます。gen3の結果と学習条件は末尾の公開gen3節、以下の過去結果は各世代の記録です。PPOは盤面・残駒・正確な配置セルから全合法手を直接評価するネット単体です。終局勝敗で自己対戦学習し、探索を付けていません。固定heuristic事前値も使います。3seedは同じ初期重みと学習条件を共有し、訓練乱数が異なります。

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

推論checkpoint（旧2048）は `artifacts/ppo-2048/seed-{1401,1402,1403}.pt`。`weights_only=True`で読み込み、学習履歴を参照しません。export元hashはmanifest.json。CPU/CUDAで元checkpointとの入力・logits/value・greedy選択bit一致を確認しました。

## 評価の読み方（旧世代2048）

旧実験のfresh96prefixでは3seed平均で旧6ms探索に得点差+2.104 [95%CI +1.272,+2.891]。今回別の96prefixで固定2048を再確認し、100ms教師とseed1401再開元512も比較しました。得点差CIは2座席と3seedを同prefix内で平均したcluster bootstrapで、更新seed母集団を代表するCIではありません。推論時間が違うため同時間比較でも、一般的最強の主張でもありません。

研究の詳細・訓練履歴は [research/ppo-learning-curve](https://github.com/eletim/blokus-duo/tree/research/ppo-learning-curve) および [research/ppo-4096](https://github.com/eletim/blokus-duo/tree/research/ppo-4096) を参照。mainには推論モデルと評価要約だけを追加します。

### 固定2048の別96prefixでの再確認

- 旧value6ms: point **56.08% [95%CI 51.82,60.16]**、得点差 **+0.696 [-0.195,+1.590]**。平均得点差の正の再確認は不確実。
- 固定100ms教師: point **44.97% [41.15,48.87]**、得点差 **−2.026 [−3.087,−0.988]**。教師の方が強い結果。
- 実平均: 旧6ms戦でPPO **1.218ms** / 旧探索 **4.690ms**。教師戦でPPO **1.346ms** / 教師 **70.388ms**。探索上限と実時間は異なる。
- seed1401の2048対再開元512: **147勝36敗9分、+7.536 [6.135,8.938]**。前回selected128との比較ではなく512への直接比較。

1344対局の合法性・算術・prefix単位CI・実訪問状態を監査し訓練重複0。旧6msへの改善幅には標本/seed差があり、100ms教師を上回っていない。2回の評価と今回の固定seedを保持し、良かったseedへの再選択をしない。全結果はevaluation-summary.json、試合再生と全履歴は研究branchに保存。

3seed×CPU/CUDA×26局面でexport前後の入力/logits/value/greedy選択bit一致。CPU/GPU AI対AI、人間先後の入力経路を終局まで検証した。推論ファイルは各約276KiB。

## 新世代gen2（固定3072/4096/4096）

gen2公開時のPPO CLI既定は `gen2`、seed1401でした。現在の既定は下記gen3です。旧グローバルAIデフォルトは変更しません。`gen2` は1401=3072、1402/1403=4096で、今回の評価からモデルを選び直していません。上の2048評価は旧世代の記録として保持します。

```bash
# 新世代の人間対戦（人間後手）
.venv/bin/python -m blokus.ppo_play --generation gen2 --model-seed 1401 --human-seat 1 --device cpu
# 旧2048を明示
.venv/bin/python -m blokus.ppo_play --generation 2048 --device cpu
# 世代対戦
.venv/bin/python -m blokus.ppo_play --generation gen2 --opponent ppo --opponent-generation 2048 --device cuda
```

新世代もCPU/GPUで動作するネット単体です。`artifacts/ppo-gen2/seed-{1401,1402,1403}.pt` を読み、履歴sidecarは不要。新しい96prefix・先後交換、3seed固定の再確認結果：

|相手|point［95% CI］|平均得点差［95% CI］|実平均ms（新PPO/相手）|
|---|---|---|---|
|previous|0.637 [0.6067708333333334, 0.6666666666666669]|+3.958 [3.2030815972222233, 4.6649739583333325]|1.442/1.271|
|oldvalue|0.634 [0.5963541666666664, 0.6692708333333331]|+2.918 [2.095486111111111, 3.7048611111111107]|1.286/4.583|
|teacher100|0.522 [0.4791666666666667, 0.5633680555555555]|+0.767 [-0.23658854166666635, 1.7483940972222234]|1.414/69.588|

全1728局の再生・算術・SHA・分離監査を実施。評価実訪問状態の訓練重複0。CIはprefix単位で、同一prefix内の座席とseedを独立標本扱いしません。固定モデル条件での再確認であり、同時間比較でも一般的最強の主張でもありません。教師は試合cold、学習や監査と時間計測を分離。詳細は [research/ppo-generation2](https://github.com/eletim/blokus-duo/tree/research/ppo-generation2/artifacts/ppo-generation2) と `artifacts/ppo-gen2/evaluation-summary.json`。今回追加学習はありません。

## 公開gen3: 4ch PPO、固定B64/LR3e-4/5120
PPO専用CLIの新しい既定は **gen3 / seed1751401**。このseedは成績から選ばず先頭を固定した。旧グローバルAI defaultは変更しない。旧gen2/2048は引き続き選択可能。

gen3は盤面の両者占有と両者の接続可能点の4チャネルを入力する。初手の接続可能点は元の開始点(4,4)/(9,9)であり、開始点専用入力はない。残駒と正確な候補配置も用いる。heuristic priorを持つPPOを終局勝敗で自己対戦学習したネット単体で、先読み探索なし。完全ランダム初期方策とは呼ばない。

採用元は研究commit67482585の **B**（minibatch64、LR3e-4、5120iteration）、3seed1751401/2/3。累積各81920gamesに対し、比較相手gen2は対応seed1401/2/3が49152/65536/65536games、6ch入力である。学習量・入力・batchが異なるので、性能差を4ch化だけの効果としない。

旧LR実験のgate未達は維持した後、別の事前計画で固定モデルを確認した（16e3baa）。新256prefix×3seed×先後交換のB対対応gen2は **point54.13%**（勝1/分0.5/敗0）、通常95%CI51.86–56.35%、3比較補正CI **51.37–56.87%**、平均得点差 **+1.20** [0.77,1.63]。prefix内の3seed/両席をまとめたCIであり、未知seed母集団への一般化や外部強豪への優越を証明しない。Cは採用しない。確認結果でcheckpoint/seedを選び直していない。推論CUDA/nativeは約0.5ms/手（各対戦の実測は公開audit参照）、CPUは環境依存で同速度とはしない。

```bash
python -m blokus.ppo_play --generation gen3 --model-seed 1751401 --device cpu --human-seat 0
python -m blokus.ppo_play --generation gen3 --device cuda --backend native --human-seat 1
python -m blokus.ppo_play --generation gen3 --opponent ppo --opponent-generation gen2 --device cpu
python -m blokus.ppo_play --generation 2048 --model-seed 1403 --device cpu --opponent heuristic
```
gen3のseedは1751401/2/3。旧指定1401/2/3もgen3ではそれぞれの別名として受け付ける。gen2/2048は1401/2/3のみ。seed省略時は各世代の先頭。推論exportは確認時models.jsonのSHAとbyte一致し、訓練履歴、研究module、大型状態indexを必要としない。manifestは `artifacts/ppo-gen3/manifest.json`。

### 新4ch学習と安全な再開
任意native backendのbuildは `python -m blokus.ppo_native build`（C11 compiler、なければ既存手順のziglang）。未buildでは `--backend auto` はPython fallback、`native` は明示エラー。高速化計測と同条件を使うにはnative buildが必要。
```bash
python -m blokus.ppo_runtime --output artifacts/my-gen3-training --channels 4 --backend native --device cuda --minibatch 64 --learning-rate 0.0003 --packed --iterations 64 --cap 600
python -m blokus.ppo_runtime --output artifacts/my-gen3-training --channels 4 --backend native --device cuda --packed --resume artifacts/my-gen3-training/latest.pt --iterations 128 --cap 600
```
1iterationは16games収集。minibatchは1 optimizer stepに使う局面数。新規CLIの省略値は従来16/3e-4を保持するため、推奨64は明示する。再開時は保存batch/LRを自動継承し、旧metadataなしcheckpointは16とoptimizer内LRを継承する。明示変更には `--allow-training-change` が必要で、変更前後とiterationを保存する。seed/channels/backend/packedも再開元と合わせる。CPU-onlyは `--device cpu --backend python`。

同期集約のupdate_fast/常駐packは従来16とのbit一致検証に基づく実装改善。**16→64の約1.42倍はoptimizer step数/KL停止が変わる学習条件変更であり、等価最適化ではない。** 公開Bはこのdense経路で学習・確認した。後続raggedは64で更新1.26倍/収集込み1.09倍だったが数値許容差未達のためmain未採用で、Bの棋力と混同しない。研究全体はbranch参照し巨大履歴をmainへ追加しない。

公開gen3の固定版・復元手順: [FROZEN_GEN3.md](FROZEN_GEN3.md)。
