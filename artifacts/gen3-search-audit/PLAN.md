# NN探索監査・有限比較（2026-11-07）
main/Frozen gen3/稼働Webを変更しない。固定1751401/2/3、追加学習0。
本番: SearchAgent width4, even depth2/4, budget100/500/2000msをWebから伝達。未完了反復は破棄、終局WDL、交互turnのnegamax。value学習targetは終局WDL（current/past/heuristic pool）、得点差でない。Webの棋譜はメモリとユーザdownloadのみ、保存済みbrowser-terminal-gameは自動試験で人間棋譜ではない。人間の敗因再現とは主張しない。
費用pilot5局面: 500ms序盤depth2、他4;2000ms全て4、実12〜694ms。2000msでも最大4で停止。

改善は2条件まで: B=NN候補順/価値を維持しroot12/internal4、層単位batch評価、深さ2/4の全root完成時のみ採用。C=Bに残駒合計6以下で全合法手の終局exact探索を追加。WDL単位、未完了exactはB完成結果へ戻る。モデルを変えない。batchの浮動誤差はCPU logit/value atol2e-6で検証、固定木の選択値差tol2e-6。最大batch16、次batch費用予測+余裕で停止。局面keyはboards/remaining/last/turn/passes、act内cacheのみ。

開発pilot: 新独立ゲーム5局面・500/2000msの診断、選択変更・基準候補外選択、value誤差は実継続WDLに対する診断（minimax真値ではない）。小終盤独立oracleで符号/pass/bonus/不変性を検査。
Validation: 新8prefix×先後×3seed、500ms固定。旧対単体、B対旧/B対単体、C対旧/C対単体。新prefixは4〜10ランダムply、モデルや結果で再選択しない。500msは人間実用の現設定で、同上限・実wall不一致。2000msは開発局面の費用診断のみ。
選択: 各候補の対旧/対単体point平均最大、同値B優先。gate=両者point>=0.50、対旧得点差>=0、fallback<50%、p99時間<750ms。探索的gateで成功判定ではない。未達なら確認なし、追加調整なし。
通過時fresh32prefix×先後×3seedで選択候補対旧/単体の2主比較を固定実施。主指標point(勝1分.5)、prefix内両席/3seedを集約bootstrap。通常95%および2比較Bonferroni97.5%CI、得点差95%CI。補正point下限>.5をその対照への改善確認とする。標本は小規模・暫定で、人間超えとは言わない。
計測上限60分、標本を途中成績で増減しない。超過時は未完報告。計測と監査/学習を重ねない。モデルSHA固定、全棋譜独立合法性/得点/算術監査、既存評価bank・既存訓練状態と開始状態の分離。完全訓練indexへのアクセス制限があれば範囲を明記。冷起動とwarm定常を分離。コード/結果は研究branchのみ。

実装検証時の例外（対戦validationを見る前）: CPU batched logit差最大3.8147e-6が初期atol2e-6を超過。失敗ログを保持し、logit許容値のみ1e-5へ明示変更する。valueと完成木は2e-6維持、argmax/着手一致も検査。これはbit等価の主張を撤回するもので、対戦gate変更ではない。batch構造変更の演算順差として比較実験で棋力への影響も含めて評価する。
