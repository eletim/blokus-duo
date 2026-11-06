"""Verify frozen gen3 file bytes with Python standard library only."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / 'artifacts/ppo-gen3/frozen-manifest.json').read_text())
for entry in manifest['models']:
    data = (root / entry['path']).read_bytes()
    if len(data) != entry['size_bytes'] or hashlib.sha256(data).hexdigest() != entry['sha256']:
        raise SystemExit(f"FAIL: {entry['path']}")
    print(f"OK seed={entry['seed']} bytes={len(data)} sha256={entry['sha256']}")
print('All 3 frozen gen3 inference exports match.')
