"""Validate saved records without running agents or writing source data."""
import hashlib
import json
from pathlib import Path
from blokus.console import Sources

source = Sources('/home/eletim/blokus-value-placement/artifacts', '/home/eletim/.local/share/blokus-web/games.sqlite3')
before = hashlib.sha256(source.archive.read_bytes()).hexdigest()
catalog = source.catalog()
checked = 0
for group in catalog['groups']:
    for i in sorted({0, group['count']-1}):
        replay = source.game(group['id'], i)
        assert replay['frames'][-1]['done']
        checked += 1
for game in catalog['archive']:
    source.human_game(game['id'])
assert hashlib.sha256(source.archive.read_bytes()).hexdigest() == before
assert not catalog['errors']
result = dict(evaluation_replays_checked=checked, archive_replays_checked=len(catalog['archive']),
              evaluation_games=sum(g['count'] for g in catalog['groups']),
              groups=len(catalog['groups']), summaries=len(catalog['evaluations']),
              archive_sha256=before, source_errors=catalog['errors'])
print(json.dumps(result, indent=2))
