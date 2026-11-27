"""Archive raw occupancy study artifacts in bounded parts and verify every member."""
from pathlib import Path
import json,hashlib,tarfile,tempfile,shutil
R=Path('artifacts/future-occupancy')
def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 paths=sorted([p for folder in ['games','evaluation','models'] for p in (R/folder).rglob('*') if p.is_file()]+list(R.glob('*-data.npz'))+list(R.glob('*-labels.json'))+[R/'historical.json.gz',R/'new-exclusions.json.gz'])
 manifest={str(p.relative_to(R)):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in paths}
 with tempfile.TemporaryDirectory() as tmp:
  archive=Path(tmp)/'raw.tar.gz'
  with tarfile.open(archive,'w:gz',compresslevel=3) as t:
   for p in paths:t.add(p,arcname=str(p.relative_to(R)),recursive=False)
  with tarfile.open(archive,'r:gz') as t:
   for m in t:
    h=hashlib.sha256(t.extractfile(m).read()).hexdigest();assert h==manifest[m.name]['sha256']
  parts=[]
  with archive.open('rb') as f:
   i=0
   while b:=f.read(40*1024*1024):
    p=R/f'raw.tar.gz.part{i:03d}';p.write_bytes(b);parts.append(dict(name=p.name,bytes=len(b),sha256=digest(p)));i+=1
  (R/'archive-manifest.json').write_text(json.dumps(dict(format='concatenate ordered parts then tar gzip',archive_sha256=digest(archive),parts=parts,members=manifest,all_member_contents_verified=True))+'\n')
 print('ARCHIVED',len(paths),len(parts))
if __name__=='__main__':main()
