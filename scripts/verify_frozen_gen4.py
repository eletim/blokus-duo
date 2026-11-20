"""Standard-library verification; no torch, CUDA, training history required."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
m=json.loads((root/'models/gen4/manifest.json').read_text())
for row in m['models']:
 for name in ('policy','value'):
  entry=row[name];data=(root/entry['path']).read_bytes()
  assert len(data)==entry['bytes'],entry['path']
  assert hashlib.sha256(data).hexdigest()==entry['sha256'],entry['path']
 print(f"gen4 seed {row['seed']}: policy/value SHA256 OK")
