# 事前固定計画
基点main6be347f。4ch新規初期化、既存heuristic priorを保持。float32、minibatch16、epochs2、Adam3e-4、clip.2、value.5、entropy.01、KL>.03でstep後停止、gradclip.5、terminal WDL、current50/past25/heuristic25、past4iteration交換を維持。
まずnative収集+基準updateをprofile。実装等価な同期集約を第一候補。ログ/有限判定のCPU転送を集約しKL停止を維持する。compile等はprofileと小比較で採否を判断。bit一致を第一基準、非一致方式ならlogit/value最大絶対誤差1e-5、loss1e-5、grad/更新重み1e-5、同入力行動一致99.9%以上を事前許容上限とし、別途曲線を比較する。閾値を超える方式は採用せず基準に戻す。学習条件変更は今回の本学習に混ぜない。
学習3seed1751401/1751402/1751403、各4096iteration=65536games。保存0/512/1024/2048/3072/4096。生成+更新+保存wall合計5400秒上限、超過予測時は次の共通保存点で停止、無制限追加なし。最適化計測/小pilotは計算wall1800秒以内。途中非有限は停止。lossだけで強さを判断しない。
全過去評価prefix開始状態をgit旧研究branchからSHA付きで収集し新生成prefixから除外。新validation32prefix(生成seed17541000)は全新訓練訪問状態/旧評価開始状態を除外。共通候補iteration1024/2048/3072/4096から各seed選択（利用可能な共通点のみ）、criterion=.5対同seed学習前point+.25対heuristicpoint+.25対対応gen2seedpoint、同値は若いiteration。全3seed維持。gateは3seed平均対学習前point>.5かつ対heuristic>=.5、非有限なし。gate不通過ならholdoutを止め原因報告。通過時選択固定後fresh96prefix(17561000)を同じ分離規則で生成、全seed対学習前/heuristic/gen2を先後交換。CIはprefixを単位に各prefixの3seed×両座席を平均しbootstrap。gen2を安定して上回らなければ推奨変更なし。教師/探索追加は今回必須でなく実施しない。
対戦はネットgreedy単体、gen2も公開ネット単体。実時間・訪問状態重複を後監査。避けられない空盤面と採用prefixを区別。計測は学習/監査と重ねない。学習履歴は今回新規分のみを保存、旧巨大履歴はSHA参照。

測定後の方式固定: 同一rollout・2warmup/6交互反復でbaseline update中央値0.13620秒、同期集約0.13340秒、同期集約+常駐0.12462秒。後者を本学習へ採用（同入力model/optimizer/RNG bit一致）。前回常駐単独は無効だったが、今回は同期集約との組合せで約9%の更新短縮。CUDA graphは決定的演算を有効にした現構成でbackwardのcaptureが失敗。失敗した個別演算の特定は未完。決定性を無効化して速度を得ることはせず不採用。失敗コード/ログは診断artifactに保持。compile/precision/batch変更は導入しない。更新単体9%をend-to-end速度比と呼ばず、本学習前に収集込み比較を行う。
