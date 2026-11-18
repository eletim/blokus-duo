"""Post-timing independent coordinate replay and split/label audit."""
import concurrent.futures
import placement_scale as s
import placement_experiment as e
from audit_placement import replay
from blokus.engine import State,Move,play
from blokus.ppo_runtime import state_hash
import numpy as np
import gzip,json,time,hashlib,os
R=s.R

def check_game(task):
 path,bank=task;d=json.load(gzip.open(path,'rt'));visited=set();labels=[];count=0
 games=d['games'] if bank is not None else [d]
 for g in games:
  ms=bank[g['prefix']]['moves']+g['moves'] if bank is not None else g['moves']
  assert replay(ms)==g['scores'];count+=1;z=State();prefix_length=len(bank[g['prefix']]['moves']) if bank is not None else g['prefix_length']
  if bank is not None:
   diff=g['scores'][g['seat']]-g['scores'][1-g['seat']];assert diff==g['difference'] and g['point']==(diff>0)+.5*(diff==0)
   assert len(g['stats'])==len(g['moves'])
  else:
   assert d['source_sha']==e.sha(f"artifacts/ppo-gen3/seed-{d['seed']}.pt")
   assert len(g['stats'])==len(ms)-prefix_length
  for ply,m in enumerate(ms):
   if ply>=prefix_length:
    h=state_hash(z);visited.add(h)
    if bank is None:labels.append(dict(seed=g['seed'],game=g['id'],ply=ply,turn=z.turn,hash=h,target=float(np.sign(g['scores'][z.turn]-g['scores'][1-z.turn]))))
    else:
     stat=g['stats'][ply-prefix_length];assert stat['player']==z.turn and stat['method']==(d['left'] if z.turn==g['seat'] else d['right']);assert stat['depth'] in (0,2,4,6)
   play(z,Move(*m) if m else None,validate=False)
  assert z.done
  if bank is not None:visited.add(state_hash(z))
 return count,visited,labels

def main():
 start=time.perf_counter();idx=np.load(e.INDEX,mmap_mode='r');assert hashlib.sha256(idx.tobytes()).hexdigest()=='cdd1f0f1ac7a0e722a670fb541ce904187949c6982cd03fda1fa7acad30f7b4c';ex=s.exclusions();sets={};counts={};total=0
 with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool:
  for phase in ('train','dev'):
   saved=json.loads((R/f'{phase}-labels.json').read_text());lookup={(r['seed'],r['game'],r['ply']):r for r in saved['labels']};seen=set();checked=0;games=0;excluded=0
   for count,visited,labels in pool.map(check_game,[(p,None) for p in sorted((R/'games').glob(f'{phase}-*.json.gz'))],chunksize=8):
    seen.update(visited);games+=count
    for r in labels:
     key=(r['seed'],r['game'],r['ply'])
     if r['hash'] in ex or e.contains(idx,r['hash']):assert key not in lookup;excluded+=1
     else:assert lookup[key]==r;checked+=1
   assert checked==len(lookup) and seen==set(saved['visited']) and excluded==saved['excluded'];sets[phase]=seen;counts[phase]=games;total+=games;print('AUDIT',phase,games,checked,flush=True)
  assert counts['train']==3072 and counts['dev']==192 and not sets['train']&sets['dev'];trained=sets['train']|sets['dev'];evalsets={};allstarts=set();summaries={}
  for size,phase in [(384,'curve'),(1536,'curve'),(3072,'validation'),(3072,'confirmation')]:
   d=R/f'n{size}';bank=json.loads((d/f'{phase}-bank.json').read_text())['prefixes'];seen=set();games=0
   for pr in bank:
    h=state_hash(e.state(pr['moves']));assert h==pr['state_hash'] and h not in trained|ex and not e.contains(idx,h)
   for count,visited,labels in pool.map(check_game,[(p,bank) for p in sorted(d.glob(f'{phase}-??-*.json.gz'))]):seen.update(visited);games+=count
   assert not seen&trained and not seen&ex and not any(e.contains(idx,h) for h in seen)
   # The two curve stages intentionally share one bank and may visit identical states.
   key=f'{size}-{phase}';evalsets[key]=seen;counts[key]=games;total+=games;e.R=d
   pairs=sorted({p.name.split('-')[1] for p in d.glob(f'{phase}-??-*.json.gz')})
   summaries[key]={p:e.aggregate(phase,p) for p in pairs}
   if pairs:assert summaries[key]==json.loads((d/f'{phase}-summary.json').read_text())
   for name,h in json.loads((d/'evaluation-manifest.json').read_text()).items():assert e.sha(name)==h,name
   print('AUDIT',key,games,flush=True)
  for a,b in [('384-curve','3072-validation'),('1536-curve','3072-validation'),('384-curve','3072-confirmation'),('1536-curve','3072-confirmation'),('3072-validation','3072-confirmation')]:assert not evalsets[a]&evalsets[b]
 selected=json.loads((R/'selection.json').read_text());assert selected['results']==summaries['3072-validation']
 assert counts['384-curve']==108 and counts['1536-curve']==108 and counts['3072-validation']==480
 assert counts['3072-confirmation']==(384 if selected['selected'] else 0)
 e.write(R/'audit.json',dict(pass_=True,games=total,counts=counts,summaries=summaries,seconds=time.perf_counter()-start,train_dev_overlap=0,training_evaluation_overlap=0,original_training_overlap=0));print('AUDIT COMPLETE',total,flush=True)
if __name__=='__main__':main()
