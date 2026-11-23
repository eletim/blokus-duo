from pathlib import Path
import tarfile,hashlib,json
r=Path('artifacts/gen4-candidate-scale'); out=[]
groups={}
for seed in [1751401,1751402,1751403]:
 groups[f'raw-{seed}.tar.gz']=sorted((r/'source-games').glob(f'{seed}-*.json'))+sorted((r/'train').glob(f'{seed}-*.json'))
groups['evaluation-games.tar.gz']=[p for n in [3072,12288] for c in ['Frozen','old768'] for p in sorted((r/f'n{n}'/f'vs-{c}').glob('*.json'))]
for name,paths in groups.items():
 assert paths,name
 target=r/name
 with tarfile.open(target,'w:gz',compresslevel=3) as t:
  for p in paths:t.add(p,arcname=str(p.relative_to(r)),recursive=False)
 with tarfile.open(target,'r:gz') as t:
  members=t.getmembers();assert len(members)==len(paths)
  for m,p in zip(members,paths):assert hashlib.sha256(t.extractfile(m).read()).digest()==hashlib.sha256(p.read_bytes()).digest()
 out.append(dict(file=name,count=len(paths),bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
 print(out[-1],flush=True)
(r/'archive-manifest.json').write_text(json.dumps({'verified_content':True,'archives':out},indent=2)+'\n')
