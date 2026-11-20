"""Archive new research trajectories, retaining any matching existing archive bytes."""
import tarfile,json,hashlib
from pathlib import Path
R=Path('artifacts/gpu-async512');index={}
for directory in sorted(R.rglob('*')):
 if not directory.is_dir():continue
 files=sorted(directory.glob('[0-9]*-*.json.gz'))
 if not files:continue
 archive=directory.with_name(directory.name+'-games.tar.gz');hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
 if not archive.exists():
  with tarfile.open(archive,'w:gz') as tar:
   for p in files:
    info=tar.gettarinfo(str(p),arcname=p.name);info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0
    with p.open('rb') as f:tar.addfile(info,f)
 actual={}
 with tarfile.open(archive,'r:gz') as tar:
  for member in tar:
   assert member.isfile() and member.name not in actual
   actual[member.name]=hashlib.sha256(tar.extractfile(member).read()).hexdigest()
 assert actual==hashes,f'existing archive differs; refusing overwrite: {archive}'
 index[str(archive.relative_to(R))]=dict(games=len(files),sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=hashes)
(R/'archive-index.json').write_text(json.dumps(index,indent=2)+'\n')
