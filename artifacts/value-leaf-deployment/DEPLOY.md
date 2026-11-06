# E deployment 2026-11-07
Rollback SHA: 4f13460398a2a769068af3ffd731244ca6e002db.
Service: systemctl --user restart blokus-web.service
WorkingDirectory: /home/eletim/blokus-duo-ppo-main
ExecStart: /home/eletim/blokus-duo-ai/.venv/bin/python -u -m blokus.web_play --host 100.95.30.96 --port 8765 --device cpu
Fast-forward integrates c9dddd5 only (4 files), no model/tag/default changes. This is processing efficiency, not verified playing-strength improvement. Same-commit previous CPU clone96PASS/4skip and completed-tree equivalence are retained; no heavy match rerun.
Startup failure: restore rollback SHA in service checkout, restart same service, preserve integration commit in Git.
