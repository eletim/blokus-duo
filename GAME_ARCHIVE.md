# 人間対局の自動保存

Webはサーバで確定した対局開始・人間/AI着手・強制pass・終局をSQLiteへtransactionで保存します。ブラウザを閉じても確定済み分は残ります。未完対局はunfinishedで、切断を敗北としません。新規対局は別IDです。旧画面の応答がブラウザで破棄されても、旧セッションで合法に確定した着手は旧IDに記録されます。サーバが古いworker結果を破棄した場合は記録もしません。

保存失敗時は対局を継続し、画面に警告と最後の保存済み手数を表示します。次の確定イベントで棋譜全体を再保存し復旧します。その前にプロセスが終了すると未保存分は失われるため、警告中は従来の「棋譜を保存」を利用してください。I/O待ちは最大約3秒（SQLite lock待ち）で、ディスク障害時の無停止保証はありません。過去の未保存対局は復元できません。対局再開UIはありません。

## 保存先と運用

本番: `/home/eletim/.local/share/blokus-web/games.sqlite3`（repo外）。`--data-dir` または `BLOKUS_DATA_DIR` で変更可能。既定は `~/.local/share/blokus-web`。再起動で消しません。実棋譜をGitへ追加しないでください。バックアップはSQLiteの `.backup` またはPython sqlite3.Connection.backupを使用します。自動削除はありません。

```bash
# CPU環境・repo直下で実行。読み出しCLIはTorch不要。
python -m blokus.game_archive --data-dir ~/.local/share/blokus-web list --kind human
python -m blokus.game_archive --data-dir ~/.local/share/blokus-web list --since 2026-11-07 --mode deep --kind human
python -m blokus.game_archive --data-dir ~/.local/share/blokus-web export --id GAME_ID --output /tmp/game.json
python -m blokus.game_archive --data-dir ~/.local/share/blokus-web replay --id GAME_ID
# 検証専用サーバ。実対局とは別DB/種別。
python -m blokus.web_play --host 127.0.0.1 --port 8773 --device cpu --data-dir /tmp/blokus-archive-test --record-kind test
```

記録はschema_version=1、ID/UTC日時、ルール/開始点、盤面と残駒/last/pass、手番、人間側、mode/世代/seed/モデルSHA256、起動時コードcommit、時間予算、各手とAIの完成深さ・実時間、終局得点/勝敗を含みます。氏名/IP/UAは棋譜に保存しません。自動学習・外部転送・全対局一覧Web APIはありません。既存HTTPサーバのアクセスログとは別機能です。手動downloadも引き続き利用でき、game_idで対応が取れます。

本番smokeは運用者専用 `BLOKUS_TEST_TOKEN` と同値の `X-Blokus-Test-Token` ヘッダを付けて開始しkind=testとして保存します（トークン自体は保存しません）。通常利用者は操作不要です。テスト記録を人間の弱点分析に混ぜないでください。検証用CLI/suiteのDBはtmp配下に分離します。

モデル/棋力/Frozenタグは変更しません。サービスのrollbackでも保存DBを削除しないでください。旧コードへ戻すと自動保存機能は止まりますが、既存DBは保持されます。
