# 公開・rollback計画
更新前main: 2adc36fdfdee099154a5069187dbabbaa8ca95e6
サービス: systemctl --user restart blokus-web.service
作業ディレクトリ: /home/eletim/blokus-duo-ppo-main
起動: /home/eletim/blokus-duo-ai/.venv/bin/python -u -m blokus.web_play --host 100.95.30.96 --port 8765 --device cpu
URL: http://100.95.30.96:8765
失敗時: サービス停止、稼働worktreeを上記rollback commitへdetach（未コミット変更が無いことを確認）、同unitを起動。remoteをforce/rewriteしない。
Frozen tagは変更しない。重み更新なし。学習/長期評価なし。
