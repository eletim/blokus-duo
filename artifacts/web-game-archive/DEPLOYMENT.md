# 自動棋譜保存の公開計画
rollback code SHA: 63a4518996276f6974f15d351af873079af5a2ac
現unit: transient blokus-web.service、WorkingDirectory=/home/eletim/blokus-duo-ppo-main
起動: /home/eletim/blokus-duo-ai/.venv/bin/python -u -m blokus.web_play --host 100.95.30.96 --port 8765 --device cpu
同unitのEnvironmentへBLOKUS_DATA_DIR=/home/eletim/.local/share/blokus-webを設定してrestart。検証専用ヘッダtokenはrepo外の権限制限EnvironmentFileに置く。
失敗時はunit停止、clean確認後上記commitへdetachして同unit起動。DBと設定は削除しない。remote forceなし。
実棋譜/test棋譜本体はrepoへ保存しない。検証結果の件数/成功状態のみcommit。

## 公開確認
2026-11-07 main0f6f97fへ通常FF/push後、既存transient unitのdrop-in `/home/eletim/.config/systemd/user/blokus-web.service.d/archive.conf` にEnvironmentFileを指定。repo外 `/home/eletim/.config/blokus-web/archive.env`（0600）へ保存先・検証tokenを置いた。data dirは0700。daemon-reload/restart後active。
現URL http://100.95.30.96:8765 にてdeep/policy/search/linear×人間先後、保存表示・保存手数一致PASS。専用ヘッダで全9検証対局kind=test。全件再読み出し・合法性再生PASS。実棋譜やtokenをGitへ追加していない。
新clone CPU106PASS/4skip、別process CLIexport/replay、サーバ再起動後の既存8対局保持と新対局追加、ブラウザ終局/download・保存失敗表示・古い応答破棄PASS。失敗表示はUIフラグによる描画試験、実保存失敗/復旧はunit testで確認。
不要testサーバ/CLIは終了、本番サービスのみ稼働。モデル/Frozenタグ変更なし。追記commitは検証文書のみ、起動時code_versionは0f6f97fで正しい。
