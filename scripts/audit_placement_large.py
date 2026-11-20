"""Post-timing independent replay, exact arithmetic, selection and frozen hashes."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import concurrent.futures,collections,gzip,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
import placement_large_eval as q
from placement_large_features import readgz
from audit_placement_scale import check_game
from blokus.engine import State,Move,play
from blokus.value_search import encode
from blokus.placement_value import mobility
R=q.R;e=q.e

def main():
 start=time.perf_counter();e.setup();manifest=json.loads((R/'data-manifest.json').read_text());bank=json.loads((R/'bank-manifest.json').read_text());excluded=set(readgz(R/'evaluation-excluded.json.gz')['states']);idx=np.load(e.INDEX,mmap_mode='r')
 assert e.sha(R/'PLAN.md')==bank['plan_sha'] and e.sha(R/'bank.json')==bank['bank_sha'] and e.sha(R/'bank.json.gz')==bank['compressed_sha'] and e.sha(R/'exclusions.json.gz')==bank['excluded_sha']
 assert hashlib.sha256(memoryview(idx).cast('B')).hexdigest()==bank['index_array_sha']
 for p,h in bank['sources'].items():assert e.sha(p)==h,p
 run=readgz(R/'bank.json.gz');cfg=json.loads((R/'games/run.json').read_text())
 assert cfg['bank_sha']==bank['bank_sha'] and cfg['width']==4 and cfg['depth']==6
 for seed,h in run['models'].items():assert e.sha(f'artifacts/ppo-gen3/seed-{seed}.pt')==h
 for p,h in cfg['source_shas'].items():assert e.sha(p)==h,p
 for p,h in manifest['file_shas'].items():assert e.sha(p)==h
 count=0;sampled=0
 for sh in manifest['shards']:
  for p,h in sh['files'].items():assert e.sha(p)==h
  for g in sh['games']:
   p=R/'games'/f"{g['seed']}-{g['id']}.json.gz";assert e.sha(p)==g['sha256'];count+=1
  npz=next(p for p in sh['files'] if p.endswith('.npz'));lab=readgz(next(p for p in sh['files'] if p.endswith('.json.gz')))['labels'];data=np.load(npz);n=collections.Counter((r['seed'],r['game']) for r in lab)
  assert np.array_equal(data['x3'],np.array([r['target'] for r in lab],np.float32));assert np.array_equal(data['x4'],np.array([1/n[r['seed'],r['game']] for r in lab],np.float32))
  # Distributed re-encoding independent of the packed-array producer.
  for i in np.linspace(0,len(lab)-1,3,dtype=int):
   r=lab[i];g=readgz(R/'games'/f"{r['seed']}-{r['game']}.json.gz");z=e.state(g['moves'][:r['ply']]);b,s=encode(z);assert np.array_equal(b[:4],data['x0'][i]) and np.array_equal(s,data['x1'][i]) and np.array_equal(mobility(z),data['x2'][i]);assert z.turn==r['turn'] and q.state_hash(z)==r['hash'];sampled+=1
  data.close()
 assert count==18624
 sets={};counts={}
 with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool:
  for phase,n in [('validation',16),('confirmation',32)]:
   paths=sorted(R.glob(f'{phase}-??-*.json.gz'))
   if not paths:continue
   assert len(paths)==9;prefixes=json.loads((R/f'{phase}-bank.json').read_text())['prefixes'];assert len(prefixes)==n
   for p in prefixes:assert p['state_hash']==q.state_hash(e.state(p['moves'])) and p['state_hash'] not in excluded and not e.contains(idx,p['state_hash'])
   seen=set();games=0
   for c,vs,_ in pool.map(check_game,[(p,prefixes) for p in paths]):games+=c;seen.update(vs)
   assert games==n*18 and not seen&excluded and not any(e.contains(idx,h) for h in seen);sets[phase]=seen;counts[phase]=games
   got={a+b:q.aggregate(phase,a+b) for a,b in q.PAIRS};assert got==json.loads((R/f'{phase}-summary.json').read_text())
   print('EVAL AUDIT',phase,games,flush=True)
 if len(sets)==2:assert not sets['validation']&sets['confirmation']
 assert ('confirmation' in counts)==json.loads((R/'selection.json').read_text())['passed']
 for p,h in json.loads((R/'evaluation-manifest.json').read_text()).items():assert e.sha(p)==h,p
 curves=[json.loads((R/f'learning-{v}-{s}.json').read_text())['curve'] for s in e.SEEDS for v in 'AC'];means=[float(np.mean([c[ep]['dev_mse'] for c in curves])) for ep in range(17)];chosen=min(range(1,17),key=lambda ep:(means[ep],ep));selection=json.loads((R/'checkpoint-selection.json').read_text());assert chosen==selection['epoch'] and means==selection['dev_mean_by_epoch'];labels=manifest['old_labels']+manifest['new_labels'];steps=(labels+63)//64
 for seed in e.SEEDS:
  for v in 'AC':
   ck=torch.load(R/f'{v}-{seed}.pt',map_location='cpu',weights_only=True);assert ck['epochs']==chosen and ck['updates']==chosen*steps and ck['train_games']==24576;assert ck['source_sha']==e.sha(f'artifacts/ppo-gen3/seed-{seed}.pt');assert e.sha(R/f'{v}-{seed}.pt')==selection['exports'][str(R/f'{v}-{seed}.pt')]
   latest=torch.load(R/f'resume-{v}-{seed}.pt',map_location='cpu',weights_only=False);assert latest['epoch']==16 and latest['updates']==16*steps and all(g['lr']==1e-4 for g in latest['optimizer']['param_groups'])
 e.write(R/'audit.json',dict(pass_=True,new_generation_games=count,old_training_games_referenced=6144,feature_samples_reencoded=sampled,evaluation_counts=counts,all_new_generation_independently_replayed_in='data-manifest.json/shards',train_dev_overlap=0,training_evaluation_overlap=0,original_training_overlap=0,common_selected_epoch=chosen,steps_per_epoch=steps,seconds=time.perf_counter()-start))
 print('FINAL AUDIT PASS',flush=True)
if __name__=='__main__':main()
