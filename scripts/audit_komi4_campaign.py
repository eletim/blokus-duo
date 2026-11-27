"""Post-run integrity, split, checkpoint, and actual-trajectory audit."""
import ast,hashlib,json
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
from blokus.engine import State,Move,play,SIZES
from blokus.margin_search import key
from blokus.margin_network import NetworkEvaluator,terminal_margin
R=Path('artifacts/komi4-campaign')
def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
plan=read(R/'plan.json');assert plan['tasks_sha256']==digest(R/'tasks.json')
sets={}
counts={}
for name in ['bootstrap','old-validation','fresh-train','fresh-validation','fresh-test']:
 m=read(R/(name+'-meta.json'));sets[name]={ast.literal_eval(k) for k in m['states']}
 counts[name]={k:v for k,v in m.items() if k!='states'}
intersections={a+' / '+b:len(sets[a]&sets[b]) for i,a in enumerate(sets) for b in list(sets)[i+1:]}
assert not any(intersections.values()),intersections
models={}
for p in sorted(R.glob('gen*-*.pt')):
 ck=torch.load(p,map_location='cpu',weights_only=False)
 assert ck['komi']==4 and all(torch.isfinite(v).all() for v in ck['model'].values())
 if p.stem.endswith('last'):assert all(k in ck for k in ['optimizer','numpy_rng','torch_rng','cuda_rng'])
 models[p.name]=dict(sha256=digest(p),epoch=ck['epoch'],generation=ck['generation'],resume_optimizer='optimizer' in ck)
sel=read(R/'selection.json')
for seed,r in sel['models'].items():assert models[f'gen1-{seed}.pt']['sha256']==r['sha256']
rows=[];shared={};all_learned=sets['bootstrap']|sets['old-validation']|sets['fresh-train']|sets['fresh-validation'];traces=set()
for p in sorted((R/'games').glob('*.json')):
 g=read(p);s=State();hit=0
 for ply,m in enumerate(g['moves']):
  if g['task']['split']=='comparison' and key(s) in all_learned:hit+=1
  play(s,Move(*m) if m is not None else None,validate=True)
 assert s.done and g['scores']==[s.score(i) for i in [0,1]]
 assert g['adjusted_first_margin']==s.score(0)-s.score(1)-4
 rem=[sum(SIZES[i] for i in range(21) if s.remaining[p]>>i&1) for p in [0,1]];assert rem==g['remaining']
 if hit:shared[p.name]=hit
 traces.add(g['trace_sha256']);rows.append(dict(id=p.stem,split=g['task']['split'],margin=g['adjusted_first_margin'],unplaced_advantage=rem[1]-rem[0],bonus_delta=g['scores'][0]-g['scores'][1]-(rem[1]-rem[0])))
assert len(rows)==188
# Comparison overlap is reported rather than relabelled as independent evidence.
assert not shared,shared
result=dict(komi=4,completed_games=len(rows),unique_traces=len(traces),duplicates_not_independent=len(rows)-len(traces),datasets=counts,state_intersections=intersections,final_comparison_training_state_overlap=shared,models=models,bonus_difference_games=[r for r in rows if r['bonus_delta']],audit='all saved games legal and terminal, scores/komi exact; checkpoint tensors finite; optimizer and RNG resumable')
(R/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['datasets','models','state_intersections']},indent=2))
