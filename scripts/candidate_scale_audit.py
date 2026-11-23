"""Post-timing independent replay, nesting, label, optimizer and aggregate audit."""
import json,gzip,time,hashlib
from pathlib import Path
import numpy as np
import torch
from candidate_scale_data import R,OLD,SEEDS,save
from candidate_scale_eval import aggregate
from audit_placement import replay
import placement_experiment as e
from blokus.gen4 import load,verify
from blokus.ppo_policy import PPOAgent,collate

def main():
 start=time.perf_counter();torch.set_num_threads(1);verify()
 manifest=json.loads((R/'bank-manifest.json').read_text())
 for p,h in manifest['files'].items():assert e.sha(p)==h,p
 for p,h in json.loads((OLD/'learning-manifest.json').read_text()).items():assert e.sha(p)==h,p
 bank=json.loads((R/'new-bank.json').read_text())['games'];assert len(bank)==11520
 prior=json.loads((OLD/'train-bank.json').read_text())['games'];assert len(prior)==768
 oldroots={z['state_hash'] for ph in ['train','dev','diagnosis'] for g in json.loads((OLD/f'{ph}-bank.json').read_text())['games'] for z in g['selected']};idx=np.load(e.INDEX,mmap_mode='r');seen=set();rngs=set();newlabels=0
 for g in bank:
  assert g['rng_seed'] not in rngs;rngs.add(g['rng_seed']);assert replay(g['moves'])==g['scores'];d=json.loads((R/'train'/f'{g["seed"]}-{g["id"]}.json').read_text());assert d['moves']==g['moves'] and len(d['rows'])==len(g['selected'])
  for r in d['rows']:
   s=e.state(g['moves'][:r['ply']]);h=e.state_hash(s);assert h==r['state_hash'] and s.turn==r['turn'];assert h not in seen and h not in oldroots and not e.contains(idx,h);seen.add(h);newlabels+=1
   qs=[e.raw(q.move) for q in e.legal_moves(s)];assert len({tuple(m) for m in r['roots']})==len(r['roots']);assert all(m in qs for m in r['roots']);w=r['results']['wide'];assert w['depth']==6 and not w['fallback'];assert w['move']==r['roots'][max(range(len(r['roots'])),key=w['root_scores'].__getitem__)]
 sets={};models=[]
 for n in [3072,12288]:
  stage=R/f'n{n}';rows=json.loads((stage/'train-bank.json').read_text())['games'];assert len(rows)==n;sets[n]={(g['seed'],g['id']) for g in rows}
  for seed in SEEDS:assert sum(g['seed']==seed for g in rows)==n//3
  for g in rows:
   name=f'{g["seed"]}-{g["id"]}.json';src=(OLD if g['id']<256 else R)/'train'/name;assert e.sha(src)==e.sha(stage/'train'/name)
  for p,h in json.loads((stage/'learning-manifest.json').read_text()).items():assert e.sha(p)==h,p
  epoch=json.loads((stage/'selection.json').read_text())['epoch']
  for seed in SEEDS:
   p,v=load(seed,'native');c=PPOAgent(seed,'cpu',checkpoint=stage/f'policy-{seed}.pt',backend='native');orig=p.model.state_dict();sd=c.model.state_dict();assert all(torch.equal(t,orig[k]) for k,t in sd.items() if not k.startswith('policy.'))
   curve=json.loads((stage/f'curve-{seed}.json').read_text());expected=8*((curve['train_labels']+15)//16);ck=torch.load(stage/f'models/{seed}/epoch-8.pt',map_location='cpu',weights_only=False);assert ck['updates']==expected;assert all(int(x['step'])==expected for x in ck['optimizer']['state'].values())
   changed=sum(not torch.equal(t,orig[k]) for k,t in sd.items() if k.startswith('policy.'));assert changed or epoch==0
   models.append(dict(total_games=n,seed=seed,epoch=epoch,updates=expected,sha=e.sha(stage/f'policy-{seed}.pt'),changed_policy_tensors=changed,frozen_exact=True))
 assert sets[3072]<sets[12288] and {(g['seed'],g['id']) for g in prior}<sets[3072]
 ex=set(json.load(gzip.open(R/'excluded-states.json.gz','rt'))['states']);starts={p['state_hash'] for p in json.loads((R/'evaluation-bank.json').read_text())['prefixes']};ex-=starts;checked=0
 for n in [3072,12288]:
  for control in ['Frozen','old768']:
   for path in (R/f'n{n}'/f'vs-{control}').glob('*.json'):
    g=json.loads(path.read_text());assert replay(g['prefix_moves']+g['moves'])==g['scores'];d=g['scores'][g['seat']]-g['scores'][1-g['seat']];assert d==g['difference'] and g['point']==float(d>0)+.5*(d==0)
    s=e.state(g['prefix_moves'])
    for m,st in zip(g['moves'],g['stats'],strict=True):
     h=e.state_hash(s);assert h not in ex and not e.contains(idx,h);checked+=1;assert st['player']==s.turn;e.play(s,e.Move(*m) if m else None,validate=False)
   before=json.loads((R/f'n{n}'/f'vs-{control}-summary.json').read_text());assert aggregate(n,control)==before
 save(R/'audit.json',dict(passed=True,new_games=len(bank),new_labels=newlabels,old_games_immutable=768,nested=True,models=models,evaluation_states=checked,historical_collisions=0,independent_coordinate_replay=True,seconds=time.perf_counter()-start))
 print('SCALE AUDIT PASSED',len(bank),newlabels,checked,flush=True)
if __name__=='__main__':main()
