# 公開・rollback計画
更新前main: 2adc36fdfdee099154a5069187dbabbaa8ca95e6
サービス: systemctl --user restart blokus-web.service
作業ディレクトリ: /home/eletim/blokus-duo-ppo-main
起動: /home/eletim/blokus-duo-ai/.venv/bin/python -u -m blokus.web_play --host 100.95.30.96 --port 8765 --device cpu
URL: http://100.95.30.96:8765
失敗時: サービス停止、稼働worktreeを上記rollback commitへdetach（未コミット変更が無いことを確認）、同unitを起動。remoteをforce/rewriteしない。
Frozen tagは変更しない。重み更新なし。学習/長期評価なし。

## 反映結果
2026-11-07: mainを2adc36fから8dc4613へfast-forwardしremote push。既存unitを再起動しactive、現URLでブラウザ操作を確認。deep/policy/search/linear各人間先後のAI応答PASS（live-smoke.json）。deepは500msのみ、既定policyのまま。新cloneのCPU全テスト102PASS/4skip。人間先後のCLI完走、ブラウザの回転反転/全対局/自動pass/終局/棋譜/モード切替時の古い応答破棄/別セッション確認PASS。
起動直後の最初のHTTP接続はモデルload前で未接続。その後通常起動・ブラウザ検証成功、rollback不要。
Frozen tag object50522fc401d8d5aa460bcc0e28179dc71d0f1284、target4f13460398a2a769068af3ffd731244ca6e002dbと3モデルSHA不変。
ステージサーバ/CLI/テストは終了、本番サービス以外の本作業runなし。追加学習・長期対戦評価なし。検証記録追記commitはコード変更なしで再起動不要。

再現: `python scripts/verify_frozen_gen3.py`; CPU環境で`python -m pytest -q`（全suiteの非NN nativeテストにはREADMEのnative build必要）。`PLAYWRIGHT_MODULE=/path/to/playwright node scripts/check_deep_web.cjs http://127.0.0.1:8772 /tmp/browser-check`。CLIはDEEP_SEARCH.md参照。
