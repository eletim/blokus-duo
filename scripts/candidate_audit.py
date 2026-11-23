"""Post-timing independent coordinate replay, label and aggregate checks."""
import json,time,hashlib
from pathlib import Path
import numpy as np
import torch
import placement_experiment as e
from candidate_data import R,SEEDS,save
from candidate_eval import aggregate
from audit_placement import replay
from blokus.gen4 import load,verify
from blokus.ppo_policy import PPOAgent,collate

def main():
 start=time.perf_counter();torch.set_num_threads(1);verify();counts={};sets={};rngs=set();idx=np.load(e.INDEX,mmap_mode='r')
 for phase in ('diagnosis','train','dev'):
  bank=json.loads((R/f'{phase}-bank.json').read_text())['games'];hashes=set();rows=0
  for g in bank:
   assert g['rng_seed'] not in rngs;rngs.add(g['rng_seed']);assert replay(g['moves'])==g['scores']
   d=json.loads((R/phase/f'{g["seed"]}-{g["id"]}.json').read_text());assert d['moves']==g['moves'] and len(d['rows'])==len(g['selected'])
   for r in d['rows']:
    z=e.state(g['moves'][:r['ply']]);assert e.state_hash(z)==r['state_hash'] and z.turn==r['turn'];assert not e.contains(idx,r['state_hash']);assert r['state_hash'] not in hashes;hashes.add(r['state_hash']);rows+=1
    moves=[e.raw(q.move) for q in e.legal_moves(z)];assert all(m in moves for m in r['roots']);assert len({tuple(m) for m in r['roots']})==len(r['roots'])
    wide=r['results']['wide'];assert wide['depth']==6 and wide['move']==r['roots'][max(range(len(r['roots'])),key=wide['root_scores'].__getitem__)]
  sets[phase]=hashes;counts[phase]=dict(games=len(bank),labels=rows)
 assert all(not sets[a]&sets[b] for a,b in [('train','dev'),('train','diagnosis'),('dev','diagnosis')])
 for p,h in json.loads((R/'learning-manifest.json').read_text()).items():assert e.sha(p)==h,p
 forbidden=set.union(*sets.values());evalsets={};evaluation={}
 for phase in ('validation','confirmation'):
  if not (R/f'{phase}-H-summary.json').exists():continue
  bank=json.loads((R/f'{phase}-bank.json').read_text())['prefixes'];evalsets[phase]=set()
  for pr in bank:assert pr['state_hash'] not in forbidden and not e.contains(idx,pr['state_hash'])
  for mode in (['H','policy'] if phase=='validation' else ['H']):
   for path in (R/f'{phase}-{mode}').glob('*.json'):
    g=json.loads(path.read_text());assert replay(g['prefix_moves']+g['moves'])==g['scores'];diff=g['scores'][g['seat']]-g['scores'][1-g['seat']];assert diff==g['difference'] and g['point']==float(diff>0)+.5*(diff==0)
    z=e.state(g['prefix_moves'])
    for m,st in zip(g['moves'],g['stats'],strict=True):
     h=e.state_hash(z);assert h not in forbidden;assert not e.contains(idx,h);evalsets[phase].add(h);assert st['player']==z.turn;e.play(z,e.Move(*m) if m else None,validate=False)
   before=json.loads((R/f'{phase}-{mode}-summary.json').read_text());assert before==aggregate(phase,mode);evaluation[phase+'-'+mode]=before
 if 'confirmation' in evalsets:assert not evalsets['validation']&evalsets['confirmation']
 models=[];epoch=json.loads((R/'selection.json').read_text())['epoch']
 for seed in SEEDS:
  orig,value=load(seed,'native');p=PPOAgent(seed,'cpu',checkpoint=R/f'policy-{seed}.pt',backend='native');sd=p.model.state_dict();base=orig.model.state_dict();assert all(torch.equal(v,base[k]) for k,v in sd.items() if not k.startswith('policy.'))
  dev=json.loads(next((R/'dev').glob(f'{seed}-*.json')).read_text());max_value_error=0
  for r in dev['rows']:
   z=e.state(dev['moves'][:r['ply']]);obs,_=p._observation(z)
   with torch.no_grad():_,a=p.model(*collate([obs],'cpu'));_,b=orig.model(*collate([obs],'cpu'))
   assert torch.equal(a,b)
  ck=torch.load(R/f'models/{seed}/epoch-8.pt',map_location='cpu',weights_only=False);n=json.loads((R/f'curve-{seed}.json').read_text())['train_labels'];assert ck['updates']==8*((n+15)//16)
  assert all(int(x['step'])==ck['updates'] for x in ck['optimizer']['state'].values())
  models.append(dict(seed=seed,epoch=epoch,updates=ck['updates'],sha=e.sha(R/f'policy-{seed}.pt'),changed_policy_tensors=sum(not torch.equal(v,base[k]) for k,v in sd.items() if k.startswith('policy.')),frozen_exact=True))
 save(R/'audit.json',dict(passed=True,counts=counts,models=models,evaluation=evaluation,seconds=time.perf_counter()-start,independent_coordinate_replay=True,phase_root_overlap=0,original_training_overlap=0,all_value_parameters_unchanged=True))
 print('AUDIT PASSED',counts,flush=True)
if __name__=='__main__':main()
