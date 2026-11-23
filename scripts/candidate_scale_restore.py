"""Restore archived scaling raw records without starting generation/training."""
import hashlib,json,tarfile,os
from pathlib import Path
R=Path('artifacts/gen4-candidate-scale');OLD=Path('artifacts/gen4-candidate-policy')
for row in json.loads((R/'archive-manifest.json').read_text())['archives']:
 p=R/row['file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
 with tarfile.open(p) as t:t.extractall(R,filter='data')
with tarfile.open(OLD/'raw-games.tar.gz') as t:t.extractall(OLD,filter='data')
for n in [3072,12288]:
 dest=R/f'n{n}'
 for phase in ['train','dev']:
  (dest/phase).mkdir(exist_ok=True)
  for g in json.loads((dest/f'{phase}-bank.json').read_text())['games']:
   name=f'{g["seed"]}-{g["id"]}.json';src=(OLD if phase=='dev' or g['id']<256 else R)/phase/name;out=dest/phase/name
   if out.exists():assert out.read_bytes()==src.read_bytes()
   else:os.link(src,out)
meta=json.loads((R/'exclusions-manifest.json').read_text())
with (R/'excluded-states.json.gz').open('wb') as f:
 for part in meta['parts']:
  data=(R/part['file']).read_bytes();assert hashlib.sha256(data).hexdigest()==part['sha256'];f.write(data)
assert hashlib.sha256((R/'excluded-states.json.gz').read_bytes()).hexdigest()==meta['sha256']
print('Raw data, nested links and historical exclusion union restored; no experiment launched.')
