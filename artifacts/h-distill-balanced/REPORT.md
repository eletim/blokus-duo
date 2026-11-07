# 両席抽出によるH方策蒸留：固定A/B対照

公開HとFrozen gen3を変更せず、元gen3からpolicy MLPだけを教師Hの着手分布へ蒸留した。実終局報酬による追加強化学習ではない。旧失敗/gateは163791fのまま保存。

今回の事前gate: **False**。全3seed・epoch8固定、seed選別なし。固定3seedの推論であり、任意の更新seedへの一般化は未証明。確認試験はgate通過時のみ。

## 対照と経験量

Aは実手番0のみ、Bは各段階で実手番0/1を32/32。共通256生成ゲーム/seedから各段階64適格ペアを採用し、各arm192ラベル/seed・8epoch・96Adam更新。両armの元ゲーム/段階ペアは同じ。前回の約364–369ラベル/seedとは総量が異なるため、旧失敗との直接的な因果差分にはしない。今回の主対照は同量のA/Bである。候補0/1・未完成教師・欠損片側はペアごと除外。

開発用は別96ゲーム/seedから各段階16ペアの両席、96ラベル/seed。1ゲーム複数局面は独立標本扱いしない。Bの半数はAに対して隣接する1ply後で、同段階でも局面内容の差は不可避。全seedで外生乱数列を共有するがモデルにより軌跡は異なる。A/Bラベルは半分重複。候補数・盤面内容・padding量は席に伴って変わり、同一FLOPsの比較ではない。lossは候補数ではなく局面単位平均を維持する。

教師は固定gen3/H幅4完成6ply。60秒は固定木完走の安全上限。全対象値を真のWDL/minimax値とは呼ばない。trunk/value/position/priorは凍結。元のAdam1e-4/CE+.05KL/batch16を保持。

## 実際の採用元ゲーム数

- seed1751401: A/B共通のユニーク元ゲーム93件（B=93）、段階別適格ペア {'12': {'eligible_pairs': 256, 'selected_pairs': 64}, '20': {'eligible_pairs': 248, 'selected_pairs': 64}, '28': {'eligible_pairs': 129, 'selected_pairs': 64}}
- seed1751402: A/B共通のユニーク元ゲーム96件（B=96）、段階別適格ペア {'12': {'eligible_pairs': 256, 'selected_pairs': 64}, '20': {'eligible_pairs': 249, 'selected_pairs': 64}, '28': {'eligible_pairs': 123, 'selected_pairs': 64}}
- seed1751403: A/B共通のユニーク元ゲーム93件（B=93）、段階別適格ペア {'12': {'eligible_pairs': 256, 'selected_pairs': 64}, '20': {'eligible_pairs': 249, 'selected_pairs': 64}, '28': {'eligible_pairs': 129, 'selected_pairs': 64}}

## 新validation（勝1・引分0.5のpoint）

各行12新prefix×先後×3seed=72局。同一prefix内6局をまとめてbootstrap。B/publicとB/Aを事前の2主比較familyとしてBonferroni補正。A/public・単体は参考値。過去72局を合算しない。

|比較|point|通常95%CI|2比較補正CI|平均ゲーム得点差 (95%CI)|
|---|---:|---|---|---|
|B-public|49.31%|40.28–58.33%|39.58–59.72%|+0.486 [-1.167,+2.042]|
|B-A|49.31%|40.97–56.94%|40.28–58.33%|+0.542 [-0.722,+1.819]|
|A-public|56.25%|45.83–66.67%|44.44–67.36%|+0.806 [-1.111,+2.681]|
|policy-B-public|49.31%|40.28–57.64%|38.89–59.03%|+0.542 [-0.319,+1.306]|
|policy-A-public|49.31%|41.67–56.94%|40.28–57.64%|+0.875 [+0.097,+1.708]|

### seed差・席差（記述統計）

- B-public: seed 1751401=54.17% (+0.62), 1751402=45.83% (-0.12), 1751403=47.92% (+0.96); 席 0=43.06%, 1=55.56%
- B-A: seed 1751401=45.83% (-0.38), 1751402=39.58% (-0.50), 1751403=62.50% (+2.50); 席 0=41.67%, 1=56.94%
- A-public: seed 1751401=66.67% (+1.67), 1751402=52.08% (+0.12), 1751403=50.00% (+0.62); 席 0=47.22%, 1=65.28%
- policy-B-public: seed 1751401=56.25% (+1.75), 1751402=41.67% (+0.00), 1751403=50.00% (-0.12); 席 0=48.61%, 1=50.00%
- policy-A-public: seed 1751401=58.33% (+2.04), 1751402=37.50% (+0.50), 1751403=52.08% (+0.08); 席 0=55.56%, 1=43.06%

## 実時間・完了率

|比較/側|平均ms|p99ms|maxms|500ms超過/手数|6ply完成率(非pass)|非pass fallback|
|---|---:|---:|---:|---|---:|---:|
|B-public/0|243.53|501.08|501.64|163/970|81.0%|0|
|B-public/1|245.42|501.08|524.57|177/968|79.2%|0|
|B-A/0|241.09|500.93|501.45|164/987|81.3%|0|
|B-A/1|242.14|501.05|546.49|168/990|80.7%|0|
|A-public/0|243.46|500.92|501.50|168/975|80.6%|0|
|A-public/1|243.22|501.02|543.74|165/968|80.6%|0|

両者500ms上限、実wall一致ではない。forcedpassは元statsでfallback=Trueになるため非pass失敗と分けた。単体とHは異なる計算量。各act内cacheはreset、各試合初回も計測に含む。初回モデルロードは対局wallに含めない。

## 両席dev診断

|arm/seed|教師一致 before→after|greedy変更|教師選択top4脱落|top4重複|
|---|---|---:|---:|---:|
|A-1751401|47.92→48.96%|7.29%|1/96|96.88%|
|B-1751401|47.92→48.96%|8.33%|1/96|96.35%|
|A-1751402|55.21→54.17%|1.04%|2/96|97.14%|
|B-1751402|55.21→54.17%|1.04%|2/96|96.09%|
|A-1751403|39.58→39.58%|3.12%|5/96|94.01%|
|B-1751403|39.58→41.67%|6.25%|6/96|94.01%|

A全288devラベル:教師一致 47.57→47.57%、教師選択top4脱落8/288。診断値であり棋力のCIではない。

B全288devラベル:教師一致 47.57→48.26%、教師選択top4脱落9/288。診断値であり棋力のCIではない。

非policy tensorはbit一致。CPU/CUDAそれぞれのdevice内で更新前後のvalueがbit一致（CPUとCUDA相互のbit一致を意味しない）。教師一致率だけで強化成功とは判定しない。席偏りが旧失敗の主因だったとは未証明。教師幅4の候補範囲と凍結valueの誤差を再生産する限界がある。更新したpolicyはrootだけでなく探索内部の候補選別も変えるため、root教師一致の改善が固定木の改善と同じではない。

## 費用・検証

pilot+新データ生成wall 1894.39秒、6学習run cache/update/metrics/save 3.74秒、主要処理区間の合計31.64分（90分枠内）。Python起動・一部dev cache等の境界費用はこの成分和に含まない。監査・対戦wallは別。生成はCPU、policy更新はCUDA float32。8epoch/96steps perarm/seed、計576updates。学習・対戦・監査は重ねない。

監査: 1068生成対局と評価全対局の独立座標合法性/終局得点、全教師ラベル合法性/tie/score整合、32固定木再生、訓練分離、モデルSHA、集計再計算PASS。以前の原訓練indexはSHA参照、巨大履歴複製なし。新visited状態と評価前後の分離はaudit.json。別processCUDA12updates vs5+save/load+7でhead/Adam/NumPy/Torch/CUDA RNG一致。さらに実A/Bデータで各20updates vs7+save/load+13（epoch境界越え）も一致。全116testsPASS。CPU-only PyTorch2.7.1+cpuで6exportをロードし合法着手/value有限性、B+H自己対局完走、人間先後入力のEOF終了経路を確認（実人間対局ではない）。

対戦計測wall（全5比較）: 24.57分。全360評価対局の独立再生PASS。訓練状態・過去評価開始状態との評価前後の重複0。

## 判断と限界

事前gate未達のため独立確認は実行していない。B/publicとB/Aのpointが50%付近でも、これは劣化・無効果・同等性の証明ではない。CIが広く、旧失敗の主因が席偏りだったとも断定できない。A/publicの点推定が高くても探索的な参考結果で、Aを後付けで候補へ差し替えない。今回のvalidationだけで一般的優越・同等性・人間への強さを主張しない。本番Hを上回る候補として採用しない。席抽出の修正は実装上妥当でも、それが棋力向上を保証するとはいえない。小さな蒸留条件の救済探索はここで止める。

次の仮説は、固定または管理された探索後の実ゲーム終局WDLでpolicy/valueを更新し、探索葉と学習分布のずれを検証すること。教師手一致ではなく勝敗へ直接接続し、非探索対照/探索付き同予算の両方を比較する必要がある。例えば固定Hを対照相手にし、学習policy自身が確率的に選んだ手のlog-probと実終局報酬でon-policy更新を検証する。Hが上書きした着手を元policyのサンプルとして無補正PPOへ投入してはならない。探索後の訪問状態でのvalue校正と単体/同予算Hの勝率を別々に検査する。これは今回未実施で、効果は未証明。

人間棋譜は今回取得/利用しておらず、人間の敗因を再現したとはいわない。main/Web/Frozenは変更なし。推論export A/Bは研究名で保存し履歴sidecar不要。

## 再現

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export H_BALANCED_DIR=/tmp/h-balanced-reproduction  # new, empty output directory
python scripts/h_balanced.py pilot --games 4
python scripts/h_balanced.py train --games 256
python scripts/h_balanced.py dev --games 96
python scripts/h_balanced.py prepare
python scripts/h_balanced_train.py
python scripts/h_balanced_eval.py
python scripts/audit_h_balanced.py
python scripts/report_h_balanced.py
python scripts/play_h_distill.py --checkpoint "$H_BALANCED_DIR/B-1751401.pt" --model-seed 1751401 --search --human-seat 0
```

生成は既存出力を上書きしない。再実行時はH_BALANCED_DIRで新しい空の出力パスを使う。元gen3 3ファイルと旧訓練indexをmodels.json/pre-evaluation-separation.jsonのSHAで照合。再生成/分離監査には既存の外部index `/tmp/ppo-lr64-training.npy`（raw SHA `cdd1f0f1ac7a0e722a670fb541ce904187949c6982cd03fda1fa7acad30f7b4c`）が必要で、cloneだけで全履歴監査を再構築できるとは主張しない。通常CPU推論にはindex不要。CUDA学習PyTorch2.7.1+cu128、GPU RTX5090、CPU thread1。
