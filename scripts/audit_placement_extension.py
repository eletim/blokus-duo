"""Independent replay, numeric aggregation, disjointness and real-update audit."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import concurrent.futures,collections,gzip,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
import placement_extension_eval as q
from placement_large_features import readgz
from audit_placement_scale import check_game
from blokus.value_search import encode
from blokus.placement_value import mobility
R=q.R;e=q.e

def main():
 start=time.perf_counter();e.setup();bm=json.loads((R/'bank-manifest.json').read_text());excluded=set(readgz(R/'evaluation-excluded.json.gz')['states']);idx=np.load(e.INDEX,mmap_mode='r');da=json.loads((R/'data-audit.json').read_text())
 assert da['pass_'] and da['counts']=={'train':24576,'dev':192}
 for p,key in [('PLAN.md','plan_sha'),('bank.json','bank_sha'),('bank.json.gz','compressed_sha'),('exclusions.json.gz','excluded_sha')]:assert e.sha(R/p)==bm[key]
 assert hashlib.sha256(memoryview(idx).cast('B')).hexdigest()==bm['index_array_sha']
 for p,h in bm['sources'].items():assert e.sha(p)==h
 bank=readgz(R/'bank.json.gz');cfg=json.loads((R/'games/run.json').read_text());assert cfg['bank_sha']==bm['bank_sha'] and cfg['width']==4 and cfg['depth']==6
 assert len({g['rng_seed'] for g in bank['games']})==24768
 for seed,h in bank['models'].items():assert e.sha(f'artifacts/ppo-gen3/seed-{seed}.pt')==h
 for p,h in cfg['source_shas'].items():assert e.sha(p)==h
 count=0;sampled=0
 for sh in da['shards']:
  for p,h in sh['files'].items():assert e.sha(p)==h
  for g in sh['games']:assert e.sha(R/'games'/f"{g['seed']}-{g['id']}.json.gz")==g['sha256'];count+=1
  npz=next(p for p in sh['files'] if p.endswith('.npz'));lab=readgz(next(p for p in sh['files'] if p.endswith('.json.gz')))['labels'];data=np.load(npz);n=collections.Counter((r['seed'],r['game']) for r in lab)
  assert np.array_equal(data['x3'],np.array([r['target'] for r in lab],np.float32));assert np.array_equal(data['x4'],np.array([1/n[r['seed'],r['game']] for r in lab],np.float32))
  for i in np.linspace(0,len(lab)-1,3,dtype=int):
   r=lab[i];g=readgz(R/'games'/f"{r['seed']}-{r['game']}.json.gz");z=e.state(g['moves'][:r['ply']]);b,s=encode(z);assert np.array_equal(b[:4],data['x0'][i]) and np.array_equal(s,data['x1'][i]) and np.array_equal(mobility(z),data['x2'][i]);assert z.turn==r['turn'] and q.q.state_hash(z)==r['hash'];sampled+=1
  data.close()
 assert count==24768
 sets={};counts={};selections={};totalupdates=0
 with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool:
  for stage,phase,n,pairs in [(36864,'validation',8,['AP','AF']),(49152,'validation',8,['AP','AF']),(49152,'confirmation',64,['AQ','AR','AF','AC'])]:
   root=R/f'n{stage}';paths=sorted(root.glob(f'{phase}-??-*.json.gz'));assert len(paths)==3*len(pairs);prefixes=json.loads((root/f'{phase}-bank.json').read_text())['prefixes'];assert len(prefixes)==n
   for p in prefixes:assert p['state_hash']==q.q.state_hash(e.state(p['moves'])) and p['state_hash'] not in excluded and not e.contains(idx,p['state_hash'])
   seen=set();games=0
   for c,vs,_ in pool.map(check_game,[(p,prefixes) for p in paths]):games+=c;seen.update(vs)
   assert games==n*6*len(pairs) and not seen&excluded and not any(e.contains(idx,h) for h in seen)
   for prior in sets.values():assert not seen&prior
   key=f'{stage}/{phase}';sets[key]=seen;counts[key]=games
   got={pair:q.aggregate(stage,phase,pair,len(pairs)) for pair in pairs};assert got==json.loads((root/f'{phase}-summary.json').read_text())
   for p,h in json.loads((root/f'{phase}-models.json').read_text()).items():assert e.sha(p)==h
 for stage in (36864,49152):
  root=R/f'n{stage}';m=json.loads((root/'data-manifest.json').read_text());assert m['training_games']==stage and m['dev_games']==192
  for p,h in m['file_shas'].items():assert e.sha(p)==h
  curves=[json.loads((root/f'learning-{v}-{s}.json').read_text())['curve'] for s in e.SEEDS for v in 'AC'];means=[float(np.mean([c[ep]['dev_mse'] for c in curves])) for ep in range(17)];chosen=min(range(1,17),key=lambda ep:(means[ep],ep));sel=json.loads((root/'checkpoint-selection.json').read_text());assert chosen==sel['epoch'] and means==sel['dev_mean_by_epoch'];steps=(m['old_labels']+m['new_labels']+63)//64
  for seed in e.SEEDS:
   for v in 'AC':
    ck=torch.load(root/f'{v}-{seed}.pt',map_location='cpu',weights_only=True);assert ck['epochs']==chosen and ck['updates']==chosen*steps and ck['train_games']==stage and ck['source_sha']==e.sha(f'artifacts/ppo-gen3/seed-{seed}.pt');assert e.sha(root/f'{v}-{seed}.pt')==sel['exports'][str(root/f'{v}-{seed}.pt')]
    latest=torch.load(root/f'resume-{v}-{seed}.pt',map_location='cpu',weights_only=False);assert latest['epoch']==16 and latest['updates']==16*steps and all(g['lr']==1e-4 for g in latest['optimizer']['param_groups']);totalupdates+=latest['updates']
  selections[str(stage)]=dict(epoch=chosen,steps_per_epoch=steps,updates_per_model=16*steps)
 final=json.loads((R/'n49152/confirmation-summary.json').read_text());passed=all(v['point_adjusted_ci'][0]>.5+1e-9 and v['difference']>0 and v['stats']['A']['over_550']<=.01 for v in final.values());assert passed==json.loads((R/'n49152/decision.json').read_text())['passed']
 e.write(R/'audit.json',dict(pass_=True,new_generation_games=count,old_training_games_referenced=24576,feature_samples_reencoded=sampled,evaluation_counts=counts,train_dev_overlap=0,training_evaluation_overlap=0,original_training_overlap=0,stage_selections=selections,total_real_updates=totalupdates,seconds=time.perf_counter()-start))
 print('EXTENSION AUDIT PASS',flush=True)
if __name__=='__main__':main()
