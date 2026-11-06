"""Coordinate-set replay + score + aggregate + frozen-bank audit, after timing."""
import sys,json,hashlib,importlib.util
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from blokus.engine import Move,PIECES
from run_deeper import R,SEEDS,aggregate,write,sha
spec=importlib.util.spec_from_file_location('coordinate_oracle',Path('artifacts/secretary-review/blokus_reference.py'))
ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)

def replay(moves):
 own=[set(),set()];remaining=[set(range(21)),set(range(21))];last=[None,None];turn=0;passes=0
 for raw in moves:
  assert passes<2
  if raw is None:
   # Independent coordinate legality for all remaining pieces, including pass.
   assert not any(ref.legal_cells(PIECES[i],own[turn],own[1-turn],ref.STARTS[turn]) for i in remaining[turn])
   passes+=1
  else:
   m=Move(*raw);assert m.piece in remaining[turn]
   assert ref.valid(m.cells,own[turn],own[1-turn],ref.STARTS[turn])
   own[turn].update(m.cells);remaining[turn].remove(m.piece);last[turn]=m.piece;passes=0
  turn=1-turn
 assert passes==2
 return [-sum(len(PIECES[i]) for i in rem) if rem else (20 if last[p]==0 else 15) for p,rem in enumerate(remaining)]

def main():
 models=json.loads((R/'models.json').read_text());starts=set();counts={};files=0
 for seed,m in models.items():assert m['sha256']==sha(m['path'])
 for phase in ('validation500','confirmation500'):
  bankfile=R/f'{phase}-bank.json';bank=json.loads(bankfile.read_text())['prefixes']
  for p in bank:assert p['state_hash'] not in starts;starts.add(p['state_hash'])
  n=0
  for path in sorted(R.glob(phase+'-*-*-*.json')):
   row=json.loads(path.read_text());assert row['bank_sha256']==sha(bankfile);assert row['model_sha256']==models[str(row['seed'])]['sha256']
   assert len(row['games'])==2*len(bank)
   assert {(g['prefix'],g['seat']) for g in row['games']}=={(p['id'],s) for p in bank for s in (0,1)}
   for g in row['games']:
    scores=replay(bank[g['prefix']]['moves']+g['moves']);assert scores==g['scores'];assert g['difference']==scores[g['seat']]-scores[1-g['seat']]
    assert len(g['timings'])==len(g['moves'])
   n+=len(row['games']);files+=1
  counts[phase]=n
 for file,phase in [('validation-summary.json','validation500'),('confirmation-summary.json','confirmation500')]:
  p=R/file
  if p.exists():
   for pair,result in json.loads(p.read_text()).items():
    a,b=pair.split('-');assert result==aggregate(phase,a,b)
 diagnostic=json.loads((R/'diagnostic.json').read_text());dbank=json.loads((R/'diagnostic-bank.json').read_text())['prefixes'];dc=0
 for row in diagnostic['records']:
  prefix=dbank[row['game']]['moves'];seat=len(prefix)%2
  for values in row['outcomes'].values():
   for v in values:
    scores=replay(prefix+[row['move']]+v['moves']);assert scores==v['scores'];diff=scores[seat]-scores[1-seat];assert diff==v['difference'];assert int(np.sign(diff))==v['wdl'];dc+=1
 write(R/'coordinate-audit.json',dict(pass_=True,coordinate_legal_score_replays=counts,files=files,disjoint_new_starts=len(starts),aggregate_exact_recalculation=True,model_hashes=True,diagnostic_replays=dc))
if __name__=='__main__':main()
