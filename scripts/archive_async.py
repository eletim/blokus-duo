"""Archive only newly generated research trajectories; keep original bytes."""
import tarfile,json,hashlib,gzip
from pathlib import Path
R=Path('artifacts/gpu-async512');index={}
for directory in sorted(R.rglob('*')):
 if not directory.is_dir() or directory.name=='restart':continue
 files=sorted(directory.glob('[0-9]*-*.json.gz'))
 if not files:continue
 archive=directory.with_name(directory.name+'-games.tar.gz');hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
 with tarfile.open(archive,'w:gz') as tar:
  for p in files:
   info=tar.gettarinfo(str(p),arcname=p.name);info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0
   with p.open('rb') as f:tar.addfile(info,f)
 with tarfile.open(archive,'r:gz') as tar:
  for member in tar:
   assert member.name in hashes and hashlib.sha256(tar.extractfile(member).read()).hexdigest()==hashes[member.name]
 index[str(archive.relative_to(R))]=dict(games=len(files),sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=hashes)
# Save the resumed output as a separate small research archive, not a copy of old histories.
directory=R/'restart'
if directory.exists():
 files=sorted(directory.glob('[0-9]*-*.json.gz'));archive=R/'restart-games.tar.gz'
 with tarfile.open(archive,'w:gz') as tar:
  for p in files:tar.add(p,arcname=p.name)
 index[archive.name]=dict(games=len(files),sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
(R/'archive-index.json').write_text(json.dumps(index,indent=2)+'\n')
