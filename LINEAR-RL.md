# 非ニューラル終局学習（実験的な追加CLI）

mainへの統合判断用候補。既存default/Web/gen3は変更しない。NumPyのREINFORCEで最終勝敗から方策係数を学び、別の線形価値を最終勝敗から学ぶ。heuristic priorあり、NN教師なし。旧30係数/new39係数とも3seed各16384games。

未使用128prefix・先後・3seed、各組768局の確認で新方策/学習前point64.71%、新/旧30係数45.25%、新/gen3 19.01%。新価値6ms上限探索/gen3 35.87%（95%CI32.81–39.00）、平均得点差-3.45。gen3は固定3seed1751401/2/3の先読みなし公開PPO。pointは勝ち1/引分0.5。探索実2.40ms、gen3 .315msで同消費時間比較ではない。gen3を置き換える根拠はない。モデルは全て履歴不要JSON、パラメータを推論時に更新しない。

```bash
python -m blokus.linear_play2 --checkpoint models/linear-value/new-3101.json --backend python --human-seat 0
python -m blokus.linear_play2 --checkpoint models/linear-value/new-3101.json --backend native --search-ms 6 --opponent-checkpoint models/linear-value/old-3101.json
```

nativeは既存READMEに従いビルドする。Python fallbackは依存requirementsのみで動く。候補限定root6/reply3/depth2、全root完了時のみ採用、時間切れは単体へ戻すsoft deadline。初回初期化は別途費用あり。研究branch `research/linear-value-features` の `artifacts/linear-value/REPORT.md` に事前計画・全結果・再現・監査。研究履歴/巨大索引は推論に不要。

nativeビルドは `python -m blokus.ppo_native build`。Cコンパイラがない環境では任意依存ziglangのインストールが必要。CPU-only venvにコンパイラもziglangもない場合はビルドできないが、Python fallbackは利用できる。今回は開発venvのziglangでビルド後、CPU-only推論を確認した。
