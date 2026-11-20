"""Independent coordinate replay and post-timing separation/statistics audit."""
import importlib.util
from functools import lru_cache
from placement_experiment import *
from blokus.engine import PIECES
spec=importlib.util.spec_from_file_location('reference','artifacts/secretary-review/blokus_reference.py');oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)
@lru_cache(maxsize=16384)
def can_play(own,other,pieces,turn):
 return any(oracle.legal_cells(PIECES[i],set(own),set(other),oracle.STARTS[turn]) for i in pieces)
def replay(ms):
 own=[set(),set()];remaining=[set(range(21)),set(range(21))];last=[None,None];turn=0;passes=0
 for row in ms:
  assert passes<2
  if row is None:
   assert not can_play(tuple(sorted(own[turn])),tuple(sorted(own[1-turn])),tuple(sorted(remaining[turn])),turn);passes+=1
  else:
   m=Move(*row);assert m.piece in remaining[turn] and oracle.valid(m.cells,own[turn],own[1-turn],oracle.STARTS[turn]);own[turn].update(m.cells);remaining[turn].remove(m.piece);last[turn]=m.piece;passes=0
  turn=1-turn
 assert passes==2
 return [-sum(len(PIECES[i]) for i in rem) if rem else (20 if last[p]==0 else 15) for p,rem in enumerate(remaining)]
def audit():
 setup();ex=forbidden();start=time.perf_counter();idx=np.load(INDEX,mmap_mode='r');assert hashlib.sha256(idx.tobytes()).hexdigest()=='cdd1f0f1ac7a0e722a670fb541ce904187949c6982cd03fda1fa7acad30f7b4c';sets={};games=0;rows={}
 for phase in ('train','dev'):
  seen=set();labels=json.loads((R/f'{phase}-labels.json').read_text());lookup={(r['seed'],r['game'],r['ply']):r for r in labels['labels']};count=0
  for seed in SEEDS:
   d=json.load(gzip.open(R/f'{phase}-{seed}.json.gz','rt'));assert d['source_sha']==sha(f'artifacts/ppo-gen3/seed-{seed}.pt')
   for g in d['games']:
    assert replay(g['moves'])==g['scores'];games+=1;s=State()
    for ply,m in enumerate(g['moves']):
     if ply>=g['prefix_length']:
      h=state_hash(s);seen.add(h);r=lookup.get((seed,g['id'],ply))
      if r:assert r['hash']==h and r['turn']==s.turn and r['target']==float(np.sign(g['scores'][s.turn]-g['scores'][1-s.turn])) and not contains(idx,h) and h not in ex;count+=1
     play(s,Move(*m) if m else None,validate=False)
  assert count==len(labels['labels']) and seen==set(labels['visited']);sets[phase]=seen;print('audit',phase,games,flush=True)
 assert not sets['train']&sets['dev'];trained=sets['train']|sets['dev'];starts=set();pilot={state_hash(State(**s)) for s in json.loads((R/'cost-pilot.json').read_text())['states']};assert not trained&pilot
 for phase in ('validation','fixed','confirmation'):
  bank=json.loads((R/f'{phase}-bank.json').read_text())['prefixes'];seen=set();hits=0;count=0
  for pr in bank:
   h=state_hash(state(pr['moves']));assert h==pr['state_hash'] and h not in trained|starts|ex and not contains(idx,h);starts.add(h)
  for path in sorted(R.glob(f'{phase}-??-*.json.gz')):
   d=json.load(gzip.open(path,'rt'))
   for g in d['games']:
    ms=bank[g['prefix']]['moves']+g['moves'];assert replay(ms)==g['scores'];score=g['scores'];diff=score[g['seat']]-score[1-g['seat']];assert diff==g['difference'] and g['point']==(diff>0)+.5*(diff==0);games+=1;count+=1
    s=state(bank[g['prefix']]['moves']);seen.add(state_hash(s))
    for m,st in zip(g['moves'],g['stats'],strict=True):
     assert st['player']==s.turn and st['method']==(d['left'] if s.turn==g['seat'] else d['right'])
     if phase=='fixed' and not st['forced_pass']:assert st['depth']==4
     play(s,Move(*m) if m else None,validate=False);seen.add(state_hash(s))
   print('replayed',path.name,flush=True)
  for h in seen:hits+=contains(idx,h)
  assert hits==0 and not seen&trained and not seen&pilot
  rows[phase]=dict(games=count,original_training_hits=int(hits),new_training_hits=len(seen&trained));sets[phase]=seen
  pairs=sorted({p.name.split('-')[1] for p in R.glob(f'{phase}-??-*.json.gz')})
  rows[phase]['results']={p:aggregate(phase,p) for p in pairs}
 assert not sets['validation']&sets['confirmation'] and not sets['fixed']&sets['validation'] and not sets['fixed']&sets['confirmation']
 for name,h in json.loads((R/'evaluation-manifest.json').read_text()).items():assert sha(name)==h,name
 selection=json.loads((R/'selection.json').read_text());assert selection['results']==rows['validation']['results']
 write(R/'audit.json',dict(pass_=True,games=games,rows=rows,train_dev_overlap=0,evaluation_split_overlap=0,seconds=time.perf_counter()-start,source_sha={str(p):sha(p) for p in Path('artifacts/ppo-gen3').glob('seed-*.pt')}));print('AUDIT PASS',games,flush=True)
if __name__=='__main__':audit()
