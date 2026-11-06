# 実験的な非NN混成探索（公開gen3の置換ではない）

旧30係数方策と元24係数価値を固定した3seed3101/2/3。元モデルはresearch/linear-hybrid-value ec64667のexportで、元方策/価値の学習終点は1024iteration。今回学習なし。再学習Pは不使用。policy/valueはNumPy線形演算、gen3の値/着手を教師にしていない。

```bash
python -m blokus.ppo_native build
python -m blokus.linear_play2 --checkpoint artifacts/linear-hybrid/exports/seed-3101.json --backend native --completed-search --search-ms 100 --root-width 24 --reply-width 8 --depth 2 --human-seat 0
```
ビルドなしは`--backend python`、人間後手は`--human-seat 1`。CPU専用Torch+NumPyの既存依存で動作する（TorchのNNをこの方策/価値で使用しない）。推論に訓練履歴/研究評価スクリプトは不要。

完成6x3/2plyを保持した後24x8/2plyを全root完了時だけ採用。未完成なら完成狭幅、狭幅も未完成ならgreedyにfallback。幅/深さは独立指定でき、`--depth 4 --root-width 6 --reply-width 3`で反復深化条件。枝を制限した近似で完全探索ではない。値はWDL予測、得点差や厳密minimax値ではない。

別の固定128prefix・3seed・先後交換（768局/比較）で、広幅B対狭幅Aは450勝45分273敗、point61.52%、2主比較補正CI57.81–65.17%、得点差+2.09。対公開gen3（対応1751401/2/3）は231勝30分507敗、point32.03%、補正CI28.65–35.48%、得点差−4.86。gen3未達。実消費B約20ms対A約2ms、gen3約0.4msで同wall優越ではない。初回依存読込で約0.6秒、warm後も単発の時間超過を記録しておりhard deadline保証はない。

main/Web/既存既定は変更しない。本候補は統合判断用である。全結果・事前計画・監査・再現方法はresearch/linear-search-width-depthのartifacts/linear-search/を参照。全108テスト、独立座標全棋譜2688局監査、CPU CLIに加え、この最小候補worktreeで関連10テストとCPU-only AI対戦を確認済み。
