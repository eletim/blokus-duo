# 非NN L の公開統合

候補 aa503027 の3モデルと24×8/2ply/100ms完成段階fallbackを変更せず、CLIとWebが共通factoryを使う形で統合。既定はgen3、非NNの既定seedは3101。学習・長期評価は実施していない。

新clone a64650e、CPU専用torch環境で93 passed / 4 skipped (14.78秒)。元exportのSHA・特徴・価値・完成木一致を既存回帰テストで確認。CLIは人間先後とも終局まで完走。Chromium実操作で両席、合法位置、回転反転、AI応答、モード切替、新規ゲーム、遅延した旧応答破棄、セッション分離、強制pass/終局、棋譜保存を確認。スマートフォン幅390pxでも横溢れなし。

合法配置の基準点が既存駒上になる場合に緑の案内が消える表示不具合を修正。非同期通信の旧エラー/終了処理が新ゲームへ干渉しないよう世代チェックを追加。

coldのモデル読込とwarmupはHTTP受付前に分離しサーバログへ記録。新cloneでは読込117.8ms、gen3 warmup0.6ms、非NN warmup47.8ms。これは初期費用の一例であり、着手速度やhard deadline保証ではない。

独立確認のL/F point63.51%（補正CI60.71–66.24%）を採用根拠とする。L/gen3は48.86%（46.22–51.47%）、平均得点差−1.51で優越未確認。非NNはgen3単体より実思考時間が長い。今回のWeb追加は棋力の追加改善ではない。

## 再現
```
python -m pytest -q
python -m blokus.non_neural_play --model-seed 3101 --backend python --human-seat 0
python -m blokus.non_neural_play --model-seed 3101 --backend python --human-seat 1
python -m blokus.web_play --host 127.0.0.1 --port 8767 --device cpu
PLAYWRIGHT_MODULE=/path/to/playwright node scripts/check_linear_web.cjs http://127.0.0.1:8767 /tmp/browser-result
```

rollback SHA・元起動コマンド・unitは rollback.json に再起動前に保存。サービス設定とURLを保った通常FF統合・再起動を行う。反映後の確認はdeployment.json/live-browser.jsonに記録する。
