"""Archive only new games, keep every source byte and per-file hash."""
import gzip,json,tarfile
from pathlib import Path
import placement_experiment as e
R=Path('artifacts/value-placement-49152')
def main():
 out=R/'game-archives';out.mkdir(exist_ok=True);bank=json.load(gzip.open(R/'bank.json.gz','rt'));archives=[]
 for seed in e.SEEDS:
  tasks=[t for t in bank['games'] if t['seed']==seed]
  for k in range(0,len(tasks),1024):
   subset=tasks[k:k+1024];dest=out/f'{seed}-{k//1024:02d}.tar.gz';files=[R/'games'/f"{t['seed']}-{t['id']}.json.gz" for t in subset]
   if not dest.exists():
    with tarfile.open(dest,'w:gz',compresslevel=1) as tar:
     for f in files:tar.add(f,arcname=f.name)
   with tarfile.open(dest) as tar:
    assert tar.getnames()==[p.name for p in files]
    for f in files:assert tar.extractfile(f.name).read()==f.read_bytes()
   assert dest.stat().st_size<95_000_000;archives.append(dict(path=str(dest),sha256=e.sha(dest),bytes=dest.stat().st_size,files={p.name:e.sha(p) for p in files}));print('ARCHIVE',dest,flush=True)
 e.write(R/'game-archive-manifest.json',dict(archives=archives,games=len(bank['games']),source_bank_sha=e.sha(R/'bank.json.gz'),restore='extract each archive into artifacts/value-placement-49152/games; only flat known filenames'))
if __name__=='__main__':main()
