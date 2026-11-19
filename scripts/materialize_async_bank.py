"""Restore canonical bank bytes from compressed JSON; do not overwrite conflicts."""
import argparse,gzip,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('--output',type=Path);p.add_argument('--expected-sha');a=p.parse_args()
original=gzip.decompress(a.archive.read_bytes());obj=json.loads(original)
raw=(json.dumps(obj,indent=2)+'\n').encode()
if a.expected_sha:
 matches=[v for v in (original,raw) if hashlib.sha256(v).hexdigest()==a.expected_sha]
 assert matches,'neither original nor canonical bytes match expected SHA'
 raw=matches[0]
sha=hashlib.sha256(raw).hexdigest()
out=a.output or a.archive.with_suffix('')
if out.exists():assert out.read_bytes()==raw,'existing bank differs; refusing overwrite'
else:
 tmp=out.with_suffix(out.suffix+'.restore.tmp');tmp.write_bytes(raw);tmp.replace(out)
print(json.dumps(dict(path=str(out),sha256=sha,games=len(obj['games']))))
