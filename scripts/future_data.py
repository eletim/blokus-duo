from future_common import *
from blokus.future_occupancy import terminal_labels
from blokus.value_inputs import encode
from blokus.placement_value import mobility
from placement_experiment import contains,INDEX

def exclusions():
 path=R/'historical.json.gz'
 if path.exists():return set(json.load(gzip.open(path,'rt'))['states'])
 old=Path('/home/eletim/blokus-gen4-policy/artifacts/gen4-candidate-scale');src=old/'excluded-states.json.gz'
 ex=set(json.load(gzip.open(src,'rt'))['states']);sources=[dict(path=str(src),sha=sha(src))]
 for p in old.glob('n*/vs-*/*.json'):
  g=json.loads(p.read_text());s=state(g['prefix_moves'])
  for m in g['moves']:
   ex.add(state_hash(s));play(s,Move(*m) if m else None,validate=False)
 sources.append(dict(path=str(old/'evaluation-games.tar.gz'),sha=sha(old/'evaluation-games.tar.gz')))
 with gzip.open(path,'wt') as f:json.dump(dict(states=sorted(ex),sources=sources),f)
 return ex

def prepare():
 if (R/'data-manifest.json').exists():return
 torch.set_num_threads(1);start=time.perf_counter();ex=exclusions();idx=np.load(INDEX,mmap_mode='r');seen=set();seen_exact=set();manifest={}
 # Whole-game split is immutable. Any collided example is omitted, never moved between splits.
 for phase in ['pilot','train','dev','test']:
  arrays=[];labels=[];visited=[];dropped=0
  for p in sorted((R/'games'/phase).glob('*.json')):
   g=json.loads(p.read_text());s=State();local=[]
   for ply,m in enumerate(g['moves']):
    if ply>=g['prefix_length'] and not s.done:
     h=state_hash(s);c=canon(s);visited.append((h,c))
     if phase!='pilot' and h not in ex and not contains(idx,h) and c not in seen and h not in seen_exact:
      b,st=encode(s);mask=~b[:2].any(0);y=terminal_labels(g['boards'],s.turn)
      assert np.all(y[b[0].astype(bool)]==0) and np.all(y[b[1].astype(bool)]==1)
      target=float(np.sign(g['scores'][s.turn]-g['scores'][1-s.turn]));local.append((b[:4],st,mobility(s),y,mask,target,min(ply//12,2),dict(game=g['id'],ply=ply,turn=s.turn,hash=h,canonical=c)))
     else:dropped+=1
    play(s,Move(*m) if m else None,validate=True)
   assert s.done and s.boards==g['boards'] and [s.score(0),s.score(1)]==g['scores']
   for b,st,extra,y,mask,target,stage,meta in local:arrays.append((b,st,extra,y,mask,target,1/len(local),stage));labels.append(meta)
  seen.update(c for h,c in visited);seen_exact.update(h for h,c in visited)
  if phase=='pilot':continue
  assert arrays and len({x['game'] for x in labels})==len(list((R/'games'/phase).glob('*.json')))
  keys=['board','state','extra','occupancy','empty','value','weight','stage'];ds={k:np.stack([a[i] for a in arrays]) for i,k in enumerate(keys)}
  np.savez_compressed(R/f'{phase}-data.npz',**ds);save(R/f'{phase}-labels.json',labels)
  manifest[phase]=dict(games=len({x['game'] for x in labels}),positions=len(labels),dropped=dropped,data_sha=sha(R/f'{phase}-data.npz'),labels_sha=sha(R/f'{phase}-labels.json'))
  print('DATA',phase,manifest[phase],flush=True)
 with gzip.open(R/'new-exclusions.json.gz','wt') as f:json.dump(dict(exact=sorted(seen_exact),canonical=sorted(seen)),f)
 save(R/'data-manifest.json',dict(splits=manifest,seconds=time.perf_counter()-start,historical_sha=sha(R/'historical.json.gz'),index_raw_sha=hashlib.sha256(memoryview(idx).cast('B')).hexdigest(),new_exclusions_sha=sha(R/'new-exclusions.json.gz')))
if __name__=='__main__':prepare()
