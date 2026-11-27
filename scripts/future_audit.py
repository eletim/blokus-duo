"""Post-evaluation independent replay, no evaluation-driven model selection."""
from future_common import *
from future_data import exclusions
from future_train import tensors,metrics
from future_eval import PAIRS,aggregate,model
from audit_placement import replay
from placement_experiment import contains,INDEX
from blokus.future_occupancy import terminal_labels
from blokus.value_inputs import encode
from blokus.placement_value import mobility

def main():
 start=time.perf_counter();torch.set_num_threads(1);verify();plan=json.loads((R/'FINAL_PLAN.json').read_text())
 for p,h in plan['code_sha'].items():assert sha(p)==h,p
 for p,h in json.loads((R/'evaluation-manifest.json').read_text()).items():assert sha(p)==h,p
 ex=exclusions();idx=np.load(INDEX,mmap_mode='r');seenc=set();seenh=set();counts={};selected={};
 for phase in ['pilot','train','dev','test']:
  labels=json.loads((R/f'{phase}-labels.json').read_text()) if phase!='pilot' else []
  ds=np.load(R/f'{phase}-data.npz') if phase!='pilot' else None;lookup={(r['game'],r['ply']):(i,r) for i,r in enumerate(labels)};ownc=set();ownh=set();n=0
  for path in sorted((R/'games'/phase).glob('*.json')):
   g=json.loads(path.read_text());assert replay(g['moves'])==g['scores'];s=State()
   for ply,m in enumerate(g['moves']):
    if ply>=g['prefix_length']:
     h=state_hash(s);c=canon(s);ownh.add(h);ownc.add(c)
     if (g['id'],ply) in lookup:
      i,r=lookup[g['id'],ply];assert r['hash']==h and r['canonical']==c and r['turn']==s.turn
      assert h not in seenh and c not in seenc and h not in ex and not contains(idx,h)
      b,st=encode(s);assert np.array_equal(ds['board'][i],b[:4]) and np.array_equal(ds['state'][i],st) and np.array_equal(ds['extra'][i],mobility(s))
      assert np.array_equal(ds['occupancy'][i],terminal_labels(g['boards'],s.turn)) and np.array_equal(ds['empty'][i],~b[:2].any(0))
      assert ds['value'][i]==float(np.sign(g['scores'][s.turn]-g['scores'][1-s.turn]));n+=1
    play(s,Move(*m) if m else None,validate=False)
   assert s.done and s.boards==g['boards'] and g['teacher']=='Frozen gen4 search CPU500ms width4 max6'
  assert n==len(labels);counts[phase]=dict(games=len(list((R/'games'/phase).glob('*.json'))),positions=n);seenc.update(ownc);seenh.update(ownh);selected[phase]=set(r['hash'] for r in labels)
 pr=json.loads((R/'eval-bank.json').read_text());checked=0
 for left,right in PAIRS:
  pair=left+right
  for path in (R/'evaluation'/pair).glob('*.json'):
   g=json.loads(path.read_text());ms=pr[g['prefix']]['moves']+g['moves'];assert replay(ms)==g['scores'];s=state(pr[g['prefix']]['moves'])
   assert g['difference']==g['scores'][g['seat']]-g['scores'][1-g['seat']];assert g['point']==float(g['difference']>0)+.5*(g['difference']==0)
   for m,st in zip(g['moves'],g['stats'],strict=True):
    h=state_hash(s);assert h not in seenh and h not in ex and not contains(idx,h);assert canon(s) not in seenc;assert st['player']==s.turn;checked+=1;play(s,Move(*m) if m else None,validate=False)
  before=json.loads((R/f'eval-{pair}.json').read_text());assert before==aggregate(pair)
 selection=json.loads((R/'selection.json').read_text());trained=[];test=tensors('test','cpu');testmetrics={}
 for seed in SEEDS:
  for arm in 'ABC':
   name=f'{arm}-{seed}';sel=selection['models'][name];assert sel['epoch']==min(range(9),key=lambda i:sel['curve'][i]['dev']['value_mse'])
   expected=8*((counts['train']['positions']+127)//128);ck=torch.load(R/'models'/name/'epoch-8.pt',map_location='cpu',weights_only=False);assert ck['updates']==expected and all(int(v['step'])==expected for v in ck['optimizer']['state'].values())
   assert sel['total_updates']==expected
   _,ev=model(seed,arm);testmetrics[name]=metrics(ev.model,test);trained.append(dict(name=name,updates=expected,epoch=sel['epoch'],export_sha=sha(R/f'{name}.pt')))
 save(R/'test-metrics.json',testmetrics);save(R/'audit.json',dict(passed=True,counts=counts,evaluation_states=checked,models=trained,independent_coordinate_replay=True,leakage_hits=0,seconds=time.perf_counter()-start))
 print('AUDIT PASSED',counts,checked,flush=True)
if __name__=='__main__':main()
