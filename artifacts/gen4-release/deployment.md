# gen4 service deployment / rollback

Before deployment (2026-11-20): no running `blokus.web_play`, no listener8765,
no `blokus-web.service` main unit. Existing archive EnvironmentFile drop-in and
`~/.local/share/blokus-web/games.sqlite3` remain intact. This deployment restores
a persistent user service, rather than claiming an old running service was restarted.

Previous main rollback commit: `d8f50664a8fb6afaa3b6f8775e7f212e66fc14d7`.
Previous worktree: `/home/eletim/blokus-duo-ppo-main` (left untouched).
New pinned release worktree: `/home/eletim/blokus-gen4-frozen`.
Bind: `100.95.30.96:8765` (existing Tailscale scope).

Unit: `~/.config/systemd/user/blokus-web.service` (tracked example alongside this file).
Existing `blokus-web.service.d/archive.conf` loads the preexisting private
`~/.config/blokus-web/archive.env`; its contents are not committed or printed.
Human archives remain outside Git. Deployment tests carry the existing test-token header.

```bash
systemctl --user status blokus-web.service
systemctl --user stop blokus-web.service
systemctl --user start blokus-web.service
journalctl --user -u blokus-web.service -n 30 --no-pager
```

To restore previous code while preserving the archive database: stop the service,
change only WorkingDirectory in the unit to `/home/eletim/blokus-duo-ppo-main`,
then `systemctl --user daemon-reload && systemctl --user start blokus-web.service`.
The same Python, host, port and archive EnvironmentFile apply. No DB deletion or
model/tag overwrite. To return to the exact predeployment service state, stop and
disable the new main unit; keep the existing archive drop-in and data directory.

Release checkout is detached and sparse (blokus, models, tests, scripts,
artifacts/ppo-gen3 and docs); no training data/index required for service.
Build native backend once: `PYTHONPATH=. /home/eletim/blokus-duo-ai/.venv/bin/python -m blokus.ppo_native build`.
The fixed tag/commit identifies runtime code and model bytes. Test records and
startup timing are in validation.json; full training/evaluation was not rerun.
