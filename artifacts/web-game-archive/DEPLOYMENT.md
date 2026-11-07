# 自動棋譜保存の公開計画
rollback code SHA: 63a4518996276f6974f15d351af873079af5a2ac
現unit: transient blokus-web.service、WorkingDirectory=/home/eletim/blokus-duo-ppo-main
起動: /home/eletim/blokus-duo-ai/.venv/bin/python -u -m blokus.web_play --host 100.95.30.96 --port 8765 --device cpu
同unitのEnvironmentへBLOKUS_DATA_DIR=/home/eletim/.local/share/blokus-webを設定してrestart。検証専用ヘッダtokenはrepo外の権限制限EnvironmentFileに置く。
失敗時はunit停止、clean確認後上記commitへdetachして同unit起動。DBと設定は削除しない。remote forceなし。
実棋譜/test棋譜本体はrepoへ保存しない。検証結果の件数/成功状態のみcommit。
