# 最小統合候補（main未適用）
この実験は重みを変更しない。利用する元混成exportは `artifacts/linear-hybrid/exports/seed-3101.json` /3102/3103、SHAは `models.json`。再学習PやNN価値を読み込まない。

`blokus/linear_tree.py` は既存 `Agent` と `Search` APIに依存する独立モジュール。`linear_play2.py` に `--completed-search --root-width --reply-width --depth` を追加。既存オプションでは従来Searchを使う。グローバルAI・PPO・Webの既定は変更しない。今回の差分にはエンジン/学習コード/モデル変更がない。

既存 `integration/linear-value-candidate` abc41cfに、前回 ec64667 の `artifacts/linear-hybrid/integration-candidate.patch` を適用したものが前提。今回の追加候補patchは、確認結果後に用意する `integration-search.patch`（tree/CLI/独立テストと元混成3本の小型exportのみ）。研究の対戦ログ・訓練履歴・gen3モデル・評価スクリプトは推論の依存ではない。CPU環境はNumPyと既存プロジェクト依存を利用（Python特徴経路は既存aiモジュールをimportするためTorch CPUパッケージも現状必要だが、新方式の方策/価値にNNは使わない）。nativeは任意、ビルドなしなら `--backend python`。

推論例:
```bash
python -m blokus.linear_play2 --checkpoint artifacts/linear-hybrid/exports/seed-3101.json --backend native --completed-search --search-ms 100 --root-width 24 --reply-width 8 --depth 2 --human-seat 1
```

これは枝を制限するminimax近似。完成6x3/2plyから始め、全rootの24x8/2plyが完成すれば置換する。部分結果では上書きしない。値は固定線形モデルのWDL予測。minimax真値/最善手保証/同消費時間での優越/学習改善とは呼ばない。独立確認の棋力と実費はREPORTを確認して秘書が統合判断する。

未適用の最小統合branch: `integration/linear-search-candidate`、commit `4276c8af50461f13110b329973359604a22d07a0`（abc41cf上の小型推論候補、研究履歴なし）。関連10テストとCPU-only AI対戦を確認。
