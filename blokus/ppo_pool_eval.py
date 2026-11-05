"""Frozen A/B pool comparison; training indices include both branches."""
import argparse,json,gzip,os,time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from . import ppo_curve_eval as q,terminal_eval as e
from .ppo_6144_eval import match
from .ppo_generation_eval import clustered
from .engine import State,Move,play,is_legal,legal_moves
from .visited_data import state_hash
from .tournament import digest,prefix_state,read_prefixes,summarize
from .research import sha
from .value_experiment import write
R=Path('artifacts/ppo-generation-pool');BASE=Path('artifacts/ppo-6144');SEEDS=(1401,1402,1403)

def lock():
 d=json.loads((R/'lock.json').read_text())
 for group in ('gen2','resume','previous'):
  for c in d[group].values():assert sha(c['path'])==c['sha256']
 for entries in d['fixed'].values():
  for c in entries.values():assert sha(c['path'])==c['sha256']
 assert sha(e.BASE)==d['teacher']['sha256']
 return d

def freeze():
 lock();steps=[]
 for kind in ('A','B'):
  for seed in SEEDS:steps.append({int(p.stem.split('-')[-1]) for p in (R/kind/f'seed-{seed}').glob('iteration-*.pt')})
 common=set.intersection(*steps)&{6400,6656,6912,7168}
 if not common:raise RuntimeError('No common completed checkpoint; no strength comparison')
 step=max(common);selected={}
 for kind in ('A','B'):
  selected[kind]={}
  for seed in SEEDS:
   p=R/kind/f'seed-{seed}'/f'iteration-{step}.pt';selected[kind][str(seed)]=dict(path=str(p),sha256=sha(p),iteration=step)
 dest=R/'selection.json'
 if dest.exists():assert json.loads(dest.read_text())==selected
 else:write(dest,selected)
 return selected

def audit_train():
 d=lock();freeze();oldaudit=BASE/'training-audit.json';assert json.loads(oldaudit.read_text())['pass_']
 oldrows={v['seed']:v for v in json.loads(oldaudit.read_text())['seeds']};states=set();prefixes=set();checked={};reports=[];prefix_sequences={}
 forbidden={state_hash(prefix_state(p)) for p in read_prefixes(R/'forbidden-prefixes.json')['prefixes']}
 for kind in ('A','B'):
  for seed in SEEDS:
   root=R/kind/f'seed-{seed}';path=max(root.glob('iteration-*.pt'),key=lambda p:int(p.stem.split('-')[-1]));c=torch.load(path,map_location='cpu',weights_only=False)
   resume=Path(d['resume'][str(seed)]['path']);old=torch.load(resume,map_location='cpu',weights_only=False)
   assert sha(resume)==oldrows[seed]['checkpoint_sha256']
   assert c['pool']==json.loads((R/f'pool-{kind}.json').read_text())
   assert len(c['history_chain'])>=len(old['history_chain']);new=[]
   for i,entry in enumerate(c['history_chain']):
    hp=(path.parent/entry['path']).resolve()
    if str(hp) not in checked:checked[str(hp)]=sha(hp)
    assert checked[str(hp)]==entry['sha256']
    if i<len(old['history_chain']):
     prior=old['history_chain'][i];assert entry['sha256']==prior['sha256'] and entry['games']==prior['games'];assert hp==(resume.parent/prior['path']).resolve()
    else:
     part=json.load(gzip.open(hp,'rt'));assert len(part)==entry['games'];new.extend(part)
   assert len(new)==(c['iteration']-6144)*16
   prefix_sequences[kind,seed]=[digest(g['prefix']) for g in new]
   ids=set();counts=Counter();targets_count=Counter();rejections=0;incidental=set();target_n=0
   for gi,g in enumerate(new):
    iteration=6144+gi//16;slot=gi%16
    from .terminal_ppo import pool_modes
    assert g['opponent']==pool_modes(iteration,16,kind)[slot]
    assert g['learner_seat']==(slot//4)%2
    assert g['id'] not in ids;ids.add(g['id']);s=State();prefixes.add(digest(g['prefix']));targets={t['ply']:t for t in g['targets']};assert len(targets)==len(g['targets']);rejections+=g.get('prefix_rejections',0)
    for ply,row in enumerate(g['prefix']+g['moves']):
     h=state_hash(s);states.add(h);m=Move(*row) if row is not None else None
     if ply<len(g['prefix']):assert h not in forbidden
     else:
      if h in forbidden:incidental.add(h)
      learner=g['opponent']=='current' or s.turn==g['learner_seat'];assert (ply in targets)==learner
     if ply in targets:
      t=targets[ply];assert t['player']==s.turn and t['state_hash']==h;assert t['return']==float(np.sign(g['scores'][s.turn]-g['scores'][1-s.turn]));target_n+=1;targets_count[g['opponent']]+=1
      if target_n%128==0:
       qs=legal_moves(s);assert (qs[t['action']].move if qs else None)==m
     assert is_legal(s,m) if m else not legal_moves(s);play(s,m,validate=False)
     if ply+1<=len(g['prefix']):assert state_hash(s) not in forbidden
    states.add(state_hash(s));assert s.done and g['scores']==[s.score(0),s.score(1)];counts[g['opponent']]+=1
   logs=[l for l in c['logs'] if l['iteration']>6144]
   assert len(logs)*16==len(new)
   for l in logs:assert all(np.isfinite(v) for v in l['update'].values()) and l['update']['grad_norm']>0
   for name,counter in [('game_counts',counts),('state_counts',targets_count)]:assert Counter({m:sum(l[name].get(m,0) for l in logs) for m in counter})==counter
   reports.append(dict(kind=kind,seed=seed,checkpoint_sha256=sha(path),iteration=c['iteration'],new_games=len(new),game_counts=dict(counts),state_counts=dict(targets_count),prefix_rejections=rejections,incidental_old_eval_start_overlap=len(incidental)))
   print('audited',reports[-1],flush=True);del new,c,old
 for name,values in [('training-states',states),('training-prefixes',prefixes)]:
  bp=BASE/name/'index.json';prior=q.read_hash_index(bp);delta=sorted(values-prior);folder=R/name;folder.mkdir(exist_ok=True);parts=[]
  for offset in range(0,len(delta),500000):
   p=folder/f'{offset//500000:03d}.bin.gz';p.write_bytes(gzip.compress(b''.join(bytes.fromhex(v) for v in delta[offset:offset+500000]),mtime=0));parts.append(dict(path=p.name,sha256=sha(p)))
  write(folder/'index.json',dict(count=len(values|prior),parts=parts,bases=[dict(path=os.path.relpath(bp,folder),sha256=sha(bp))]))
 paired=[]
 for seed in SEEDS:
  a=prefix_sequences['A',seed];b=prefix_sequences['B',seed];n=min(len(a),len(b));assert a[:n]==b[:n];paired.append(dict(seed=seed,common_games=n,prefix_sequences_equal=True,sha256=digest(a[:n])))
 write(R/'training-audit.json',dict(pass_=True,paired_external_prefixes=paired,rows=reports,base_audit=dict(path=str(oldaudit),sha256=sha(oldaudit)),history_files=checked,note='Prior full legal/reward audit verified by checkpoint/hash reference; all new games fully replayed. Both A/B all visited states union. Action indices checked every 128 targets; every move independently legal.'))

def bank(phase):
 q.R=R
 return q.fixture(phase,1670000 if phase=='validation' else 1680000,32 if phase=='validation' else 96)

def validation():
 d=lock();selected=freeze();f=bank('validation');direct=[];fixed={'A':[],'B':[]}
 for seed in SEEDS:
  key=str(seed)
  for kind in ('A','B'):
   ck=selected[kind][key]['path']
   for name,spec in [('previous',d['previous'][key]['path']),('gen2',d['gen2'][key]['path']),('oldvalue','oldvalue'),('teacher100','teacher100')]:
    row=match(R/kind/f'seed-{seed}'/f'validation-{name}.json',ck,spec,f);fixed[kind].append(row['summary']['point_rate'])
  row=match(R/'B'/f'seed-{seed}'/'validation-A.json',selected['B'][key]['path'],selected['A'][key]['path'],f);direct.append(row['summary']['point_rate'])
 direct_mean=float(np.mean(direct));delta=float(np.mean(fixed['B'])-np.mean(fixed['A']));advance=direct_mean>=.45 and delta>=-.05
 write(R/'decision.json',dict(advance=advance,B_vs_A_point=direct_mean,fixed_opponent_B_minus_A=delta,gate='BvsA>=.45 and four-opponent B-A>=-.05; common checkpoint fixed by compute, never by performance',selection_sha256=sha(R/'selection.json')))

def holdout():
 assert json.loads((R/'decision.json').read_text())['advance'];selected=freeze();d=lock();f=bank('holdout')
 assert json.loads((R/'holdout-preflight.json').read_text())['selection_sha256']==sha(R/'selection.json')
 for seed in SEEDS:
  key=str(seed);ck=selected['B'][key]['path']
  for name,spec in [('A',selected['A'][key]['path']),('gen2',d['gen2'][key]['path']),('teacher100','teacher100')]:match(R/'B'/f'seed-{seed}'/f'holdout-{name}.json',ck,spec,f)

def audit_eval():
 d=lock();selected=freeze();q.R=R;training=q.hashes('training-states');reports={};individual={};aggregate={}
 decision=json.loads((R/'decision.json').read_text());fixed={k:[] for k in ('A','B')};direct=[]
 for seed in SEEDS:
  direct.append(json.loads((R/'B'/f'seed-{seed}'/'validation-A.json').read_text())['summary']['point_rate'])
  for kind in ('A','B'):
   for op in ('previous','gen2','oldvalue','teacher100'):fixed[kind].append(json.loads((R/kind/f'seed-{seed}'/f'validation-{op}.json').read_text())['summary']['point_rate'])
 dm=float(np.mean(direct));delta=float(np.mean(fixed['B'])-np.mean(fixed['A']))
 assert decision['B_vs_A_point']==dm and decision['fixed_opponent_B_minus_A']==delta
 assert decision['advance']==(dm>=.45 and delta>=-.05) and decision['selection_sha256']==sha(R/'selection.json')
 phases=['validation']+(['holdout'] if (R/'holdout-prefixes.json').exists() else [])
 for phase in phases:
  f=read_prefixes(R/f'{phase}-prefixes.json');lookup={p['id']:p for p in f['prefixes']};pre=json.loads((R/f'{phase}-preflight.json').read_text());assert pre['prefix_sha256']==f['sha256'] and pre['selection_sha256']==sha(R/'selection.json');assert pre['training_states_sha256']==sha(R/'training-states/index.json')
  starts={state_hash(prefix_state(p)) for p in f['prefixes']};assert not starts&training
  for src in pre['sources']:
   assert sha(src['path'])==src['sha256'];old=json.loads(Path(src['path']).read_text());assert not starts&{state_hash(prefix_state(p)) for p in old['prefixes']};assert not set(lookup)&{p['id'] for p in old['prefixes']}
  count=0;visited=set()
  for path in sorted(R.glob(f'*/seed-*/{phase}-*.json')):
   row=json.loads(path.read_text());kind=path.parent.parent.name;seed=path.parent.name.split('-')[-1]
   op=path.stem.split('-',1)[1];expected={'A':selected['A'][seed]['path'],'previous':d['previous'][seed]['path'],'gen2':d['gen2'][seed]['path'],'oldvalue':'oldvalue','teacher100':'teacher100'}[op];assert row['right']==str(expected)
   assert row['prefix_sha256']==f['sha256'] and row['left_sha256']==selected[kind][seed]['sha256'];assert sha(row['left'])==row['left_sha256'] and sha(row['right'] if Path(row['right']).is_file() else e.BASE)==row['right_sha256'];assert row['summary']==summarize(row['games']);assert row['cold_per_game']
   seats={};calls=0
   for g in row['games']:
    s=prefix_state(lookup[g['prefix_id']]);seats.setdefault(g['prefix_id'],[]).append(g['seat'])
    for v in g['moves']:
     visited.add(state_hash(s));m=Move(*v) if v else None;assert is_legal(s,m) if m else not legal_moves(s);play(s,m,validate=False);calls+=1
    visited.add(state_hash(s));assert s.done and g['scores']==[s.score(0),s.score(1)] and g['difference']==s.score(g['seat'])-s.score(1-g['seat']);count+=1
   assert len(seats)==f['count'] and all(sorted(v)==[0,1] for v in seats.values());assert calls==sum(t['calls'] for t in row['timing'])
   individual[str(path.relative_to(R))]=dict(summary=row['summary'],timing=[{k:v for k,v in t.items() if k!='all_ms'} for t in row['timing']])
  assert not visited&training
  reports[phase]=dict(games=count,unique_actual_states=len(visited),training_actual_overlap=0,adopted_start_overlap=0)
  for op in ('A','gen2','teacher100'):
   arrays=[];times=[[],[]]
   for seed in SEEDS:
    r=json.loads((R/'B'/f'seed-{seed}'/f'{phase}-{op}.json').read_text());groups={}
    for g in r['games']:groups.setdefault(g['prefix_id'],[]).append(g['difference'])
    arrays.append(np.array([[np.mean([float(z>0)+.5*(z==0) for z in groups[p['id']]]),np.mean(groups[p['id']])] for p in f['prefixes']]))
    for i in (0,1):times[i]+=r['timing'][i]['all_ms']
   val=clustered(np.mean(arrays,axis=0));val['per_seed_mean']=np.mean(arrays,axis=1).tolist();val['timing']=[dict(mean_ms=float(np.mean(ts)),p95_ms=float(np.quantile(ts,.95)),max_ms=max(ts),over_100ms_rate=float(np.mean(np.array(ts)>100)) if i==1 and op=='teacher100' else None) for i,ts in enumerate(times)];aggregate[f'{phase}-{op}']=val
 write(R/'evaluation-audit.json',dict(pass_=True,gate_recomputed=True,opponent_specs_verified=True,phases=reports,individual=individual,aggregate=aggregate,note='Prefix-unit CI averages seats and fixed seeds inside prefix. Not seed-population CI.'))
 print(aggregate,flush=True)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('mode',choices=['freeze','audit_train','validation','holdout','audit_eval']);args=a.parse_args();globals()[args.mode]()
