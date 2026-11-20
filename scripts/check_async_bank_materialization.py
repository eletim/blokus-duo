"""Verify archive restoration against every available raw bank byte hash."""
import hashlib,json,subprocess,sys,tempfile
from pathlib import Path
R=Path('artifacts/gpu-async512');rows=[]
with tempfile.TemporaryDirectory(prefix='async-bank-check-') as tmp:
 for i,p in enumerate(sorted(R.rglob('*bank*.json.gz'))):
  raw=p.with_suffix('')
  if not raw.exists():continue
  sha=hashlib.sha256(raw.read_bytes()).hexdigest()
  out=Path(tmp)/f'{i}.json'
  subprocess.run([sys.executable,'scripts/materialize_async_bank.py',str(p),'--output',str(out),'--expected-sha',sha],check=True,capture_output=True)
  assert out.read_bytes()==raw.read_bytes()
  rows.append(dict(archive=str(p),sha256=sha,bytes=raw.stat().st_size))
assert rows
(R/'bank-materialization-check.json').write_text(json.dumps(rows,indent=2)+'\n')
