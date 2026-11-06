# 実候補GPU更新: 研究候補、等価採用gate未達
固定モデルの確認試験は先に全対戦・監査を完了（commit16e3baa）。本段階の実装はその棋力結果へ寄与していない。main/公開モデル/既定経路は変更なし。最小コード候補commit87b9af0（前提は研究側ppo_gpu.update_fast、mainへ単独cherry-pickする前に依存確認が必要）。

## 実装
CPUで候補owner/destinationを用意し実候補だけembedding/local gather/policy MLP。CNN/valueは局面ごと1回。logitsのみ元dense形状へscatterし既存Categoricalの正規化/entropy/局面平均lossを保持する。segment softmaxは導入しない。残るlogits paddingとdistribution費用は省略していない。収集・推論・sampling RNG、候補順序、KL停止位置、Adam、float32、epoch/minibatch順は変更しない。旧dense resident経路を残し明示compact=Trueのみ。compile/graphなし（初期費用/recompileなし）、AMPなし。

## 同一保存rollout・同一minibatch比較
2warmup、6回交互、初期化除外。376 learner states。旧はdense resident、新は実候補をminibatchごとpack（このCPU転送費用込み）。同MBで実optimizer step数一致。異なるMB間を等updateとは呼ばない。
|MB|旧update ms|新update ms|速度比|旧/new peak MB|steps|dense/actual候補比|
|---|---|---|---|---|---|---|
|16|111.328|111.360|1.000|169.37/77.97|37|3.600|

MB16: 実候補 89207 / dense計算候補 321144、実optimizer state visits 584。新policy計算候補は実候補数、logits/distributionはdense。
|64|50.303|39.799|1.264|310.84/92.90|12|4.070|

MB64: 実候補 118216 / dense計算候補 481168、実optimizer state visits 752。新policy計算候補は実候補数、logits/distributionはdense。

## 同量収集込み
16gamesの同一初期重み/RNG、全測定のrollout input hash一致。1warmup＋4回交互。初期化除外、同期/転送/生成/pack/update込み。
|MB|旧/new wall ms|速度比|旧/new games/sec|旧/new collect ms|旧/new update ms|
|---|---|---|---|---|---|
|16|237.51/235.92|1.007|67.36/67.82|90.77/91.44|145.89/143.46|
|64|144.80/132.75|1.091|110.50/120.52|90.89/90.72|53.49/41.49|
各raw JSONにgames/sec、learner states/sec、実optimizer state visits/sec、memory、生成CPU/GPU区分。保存rolloutとE2Eの入力分布は異なるので速度数値を混ぜない。

## 分布依存性とprofile
序盤ply<12、中盤12–23、終盤24以降を別測定。1warmup＋4回交互。
- early: MB16 1.053倍, MB64 1.013倍
- middle: MB16 0.982倍, MB64 1.121倍
- late: MB16 0.991倍, MB64 0.990倍
profile（同じ64/12steps）はembedding backward device time 7.076→1.257ms、index_put_impl 5.790→3.755ms。後者はcalls24→48であり、回数よりデータ量が削減された。operatorと子kernelの時間は重複するので合計してGPU総時間とはしない。候補数の小さい終盤ではpack/dispatchの固定費が勝ち、改善なし。

## 等価性と採否
事前許容は出力/loss/grad/1-step重み各maxabs2e-5。CPU4/6chは範囲内。CUDA4chはfresh Adam1step重み4.345e-5、CUDA6chはgrad3.058e-5で未達。出力差は最大3.815e-6、value/lossは検証入力で一致。許容差を事後変更しない。
保存rolloutのrestored Adam全update後はMB16最大重み差0.00142846、MB64 0.000020266。大部分はpolicy最後の全候補共通bias（softmaxでは理論的に相殺）であり、そのほか最大はMB16 conv2 weight9.522e-6、MB64 1.49e-7。near-zero bias勾配へのAdam適応と加算順の影響が考えられるが、同等学習保証とはしない。最終biasを凍結する等の学習条件変更は行っていない。
4iteration×16gamesのCPU/CUDA短期比較で着手列/得点は一致。2iteration目以後の入力hashはold logprob/value等を含むため一致せず、重みはbit不一致。入力hashの違いを「ゲームが変わった」と混同しない。両deviceのcompact4 vs2+別process再開2でmodel/past/Adam/全RNG bit一致、4100で過去policy交換。初回CPUはCUDA checkpointのRNG直接復元を拒否したログを保存し、CPU試験は同じ重み/optimizerからCPU RNGを明示初期化して実施（cross-device exact resumeとは呼ばない）。
全95tests PASS。4/6ch合法順序/mask/pass/各局面平均loss/勾配/配列不変/finite拒否/KLstep停止を検証。短期256局全target/行動index/learner側/終局報酬/合法性を再生監査。全benchmark/phase/E2Eで対照optimizer steps一致。CPU/CUDA固定BのCLI完走ログも保存。
**採用判断: GPU経路は研究限定・自動採用しない。** 64でupdate1.26倍/E2E1.09倍は得たが、等価性の事前条件未達。長期学習品質未検証、棋力改善とは呼ばない。B推論候補の固定試験結果は別成果として秘書へ返す。

## 再現
```bash
PYTHONPATH=. .venv/bin/python scripts/bench_ppo_ragged.py
PYTHONPATH=. .venv/bin/python scripts/check_ppo_ragged.py
PYTHONPATH=. .venv/bin/python scripts/run_ppo_ragged_checks.py
PYTHONPATH=. .venv/bin/python scripts/profile_ppo_ragged.py
PYTHONPATH=. .venv/bin/python scripts/bench_ppo_ragged_e2e.py
PYTHONPATH=. .venv/bin/python scripts/audit_ppo_ragged.py
.venv/bin/python -m pytest -q
```
native buildはREADME既存手順。torch2.7.1+cu128/RTX5090。profileと計測、監査、学習pilotは逐次実施。過去巨大履歴は追加複製なし。短期pilotと必要検証checkpointのみ。

## 次案
採用前にpolicy共通biasの浮動勾配とAdam感度を切り分ける専用数値テストを優先。bias除去/凍結は等価実装とせず別条件にする。実候補表現自体はmemoryを削減するが、さらにstate/cell集約backwardを行列積へ置換する案は未計測。多数手法を追加探索せず、許容差未達を解消できる場合のみ次の有限pilotへ。公開Bの統合判断とGPU実装の採否は分離する。
