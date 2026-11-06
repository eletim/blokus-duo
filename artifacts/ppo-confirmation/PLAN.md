# 固定モデル確認とGPU実装対照
旧LR実験のgate未達は維持。本試験はvalidationの示唆を受け事前計画する別の確認試験であり旧実験holdoutではない。
B/C iteration5120全3seed固定。新256prefix、各seed先後交換。主比較B/gen2、副比較C/gen2,C/B。全3point比較にBonferroni同時98.333333% percentile bootstrap CI（20000再標本、prefix内全seed/両席維持）を用い、通常95%も保存。補正下限>.5で優越、その他未確定。非劣性は主張しない。得点CIは記述的95%。標本追加、checkpoint/seed選択なし。
過去全訓練訪問状態と過去評価開始状態を除外し新bankを評価前固定。全対局再生/算術/訪問状態/hash監査後にGPU変更へ進む。
GPU段階は別commit/branch。float32、元重み/候補順/局面平均loss/Adam/KL停止/minibatch順維持。実候補のみpolicy計算する経路を旧dense経路と比較。事前許容誤差: 出力/loss 2e-5、勾配2e-5、1更新重み2e-5。bit一致でなければ明示、短期軌跡も比較。固定rolloutで各MB16/64のstep数一致を要求。計測/小pilot30分以内、追加本学習なし。main変更なし。
