# 1回だけの再pilot（初回結果確認後・実行前固定）
初回64iterations×3seedのgateは未達（heuristic点率.5234、CI[.4635,.5833]）。損失有限・KL小・entropy約2.6、方策崩壊なし。経験量が1024games/seed、生成+更新計約112秒と小さいため、変更軸を経験量だけに限定して検証する。
常に各seedの最新64iterationから継続（validation選択checkpointからは開始しない）。モデル/初期重み/損失/PPO係数/相手割合/学習率は不変。総512iterations=8192games/seed、保存128/256/512、3seed生成+更新の累積wall上限1800秒を維持。訓練prefixは通算iterationの新seed、旧validation/holdoutは使わない。
出力artifacts/terminal-ppo/extended。新validation seed1470000、32prefix先後交換、固定heuristic/学習前poolでcheckpointを選ぶ。同一gate。通過時だけfresh holdout seed1480000、96prefix。未達ならこの依頼での修正は終了し、追加ハイパーパラメータ探索をしない。
保存実装のみ大きな履歴重複を避けてgzip圧縮（mtime0）、学習意味は不変。新保存形式でもCUDA別process再開を再検証する。
