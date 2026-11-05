# 2048再確認・配布と4096継続 事前計画
基点df7bdf8、研究branch research/ppo-4096。mainへは2048推論だけを追加し既存デフォルト維持。例示seed1401を結果より前に固定。seed選び直しなし。
第一段階: seed1570000の新96prefixを旧全訓練訪問状態3306730件/過去評価prefix・開始状態から事前除外。固定2048の3seed vs旧value6ms/固定100ms priority教師、seed1401のみ再開元512も比較。先後交換、prefix CI、全実時間と事後全訪問監査。設定調整なし。
第二段階: policy推論を履歴不要へexport。CPU/CUDA出力一致と合法性、CLI人間/AI対戦、新cloneを検証してmainへ通常cherry-pick。巨大訓練履歴は研究branchのみ。旧valueと100ms教師を残す。
第三段階: 各seed2048のoptimizer/past/RNGから4096まで継続、3072/4096保存。損失/報酬/相手割合/学習率/16ゲーム同期処理変更なし。追加生成更新IO wall合計5400秒、各seed1800秒。非有限で停止。上限到達はiteration境界で保存し無期限延長なし。
継続学習の新validation seed1590000/32prefix、各seed2048/新3072/4096を比較。選択基準は前回と同じ previous2048 point .5 + heuristic .25 + oldvalue6ms .25、同値なら同重み得点差、次に早いiteration。選択固定後seed1600000のfresh96prefixで対2048/旧6msのみ評価。すべて全訓練状態と過去評価prefix除外、先後交換、prefix内でseed平均bootstrap。再確認prefixを訓練の開始状態へ流用しない。生成/更新と時間対戦は直列。
