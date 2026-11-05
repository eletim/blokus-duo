"""Replay complete on-policy games, player-relative terminal targets and paired evaluation."""
import sys,json,time,hashlib,argparse,gzip
from pathlib import Path
from collections import Counter
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from blokus.terminal_ppo import ROOT,terminal_return,TerminalNet,observe,collate
from blokus.engine import State,Move,legal_moves
from blokus.tournament import prefix_state,read_prefixes,summarize,digest
from blokus.visited_data import state_hash
from blokus.research import sha
from blokus.value_experiment import write
from blokus.ai import seed_all
from audit_root_distillation import replay

def audit(root):
 root=Path(root);seed_all(1400);begin=time.perf_counter();train_prefixes=set();train_states=set();game_ids=set();sources=0;targets=0;initial_hashes=set();gradient_checks=[];recomputed=0
 for seed in [1401,1402,1403]:
  d=root/f'seed-{seed}';games=json.loads((d/'games.json').read_text()) if (d/'games.json').exists() else json.load(gzip.open(d/'games.json.gz','rt'));ck=torch.load(d/'untrained.pt',map_location='cpu',weights_only=False);h=hashlib.sha256()
  for k,v in ck['model'].items():h.update(k.encode());h.update(v.numpy().tobytes())
  initial_hashes.add(h.hexdigest());model=TerminalNet().cuda();model.load_state_dict(ck['model'])
  for j,g in enumerate(games):
   assert g['id'] not in game_ids;game_ids.add(g['id']);sources+=1;s=State();train_prefixes.add(digest(g['prefix']));by={r['ply']:r for r in g['targets']};assert len(by)==len(g['targets'])
   for ply,m in enumerate(g['prefix']+g['moves']):
    if ply>=len(g['prefix']):
     learner=g['opponent']=='current' or s.turn==g['learner_seat'];assert (ply in by)==learner
    if ply in by:
     r=by[ply];assert r['player']==s.turn and r['state_hash']==state_hash(s);train_states.add(r['state_hash']);qs=legal_moves(s);selected=qs[r['action']].move if qs else None;assert (None if selected is None else [selected.piece,selected.orientation,selected.row,selected.col])==m
     assert r['return']==float(np.sign(g['scores'][r['player']]-g['scores'][1-r['player']]));assert -1<=r['value']<=1;targets+=1
     if j==0 and recomputed<24:
      with torch.no_grad():l,v=model(*collate([observe(s,qs)[0]]));lp=torch.distributions.Categorical(logits=l).log_prob(torch.tensor([r['action']],device='cuda')).item()
      assert abs(lp-r['log_prob'])<2e-5 and abs(v.item()-r['value'])<2e-5;recomputed+=1
    replay(s,m)
   assert s.done and g['scores']==[s.score(0),s.score(1)]
  modes=Counter(g['opponent'] for g in games);assert modes['current']==2*modes['past']==2*modes['heuristic']
  logs=json.loads((d/'curve.json').read_text());assert all(np.isfinite(list(l['update'].values())).all() for l in logs);assert any(l['update']['grad_norm']>0 for l in logs);sample_counts={m:sum(len(g['targets']) for g in games if g['opponent']==m) for m in modes};gradient_checks.append(dict(seed=seed,games=len(games),pool=dict(modes),learning_records_by_pool=sample_counts))
 assert len(initial_hashes)==1
 fixture=read_prefixes(root/'validation-prefixes.json');assert not train_prefixes & {digest(p['moves']) for p in fixture['prefixes']};lookup={p['id']:p for p in fixture['prefixes']};evalgames=0;evalstates=set();rows=[]
 for path in root.glob('seed-*/*validation-*.json'):
  r=json.loads(path.read_text());assert r['prefix_sha256']==fixture['sha256'];assert summarize(r['games'])==r['summary'];assert sum(t['calls'] for t in r['timing'])==sum(len(g['moves']) for g in r['games'])
  for timing in r['timing']:
   assert timing['calls']==len(timing['all_ms']);assert abs(timing['mean_ms']-np.mean(timing['all_ms']))<1e-10
  for g in r['games']:
   s=prefix_state(lookup[g['prefix_id']]);evalgames+=1
   for m in g['moves']:evalstates.add(state_hash(s));replay(s,m)
   assert s.done and [s.score(0),s.score(1)]==g['scores'];assert g['difference']==s.score(g['seat'])-s.score(1-g['seat'])
  rows.append(path.name)
 selection=json.loads((root/'selection.json').read_text())
 for seed,c in selection.items():
  options=[]
  for step in (json.loads((root/'eval-config.json').read_text())['steps'] if (root/'eval-config.json').exists() else [16,32,64]):
   paths=[root/f'seed-{seed}/validation-{step}-{op}.json' for op in ['heuristic','untrained']]
   if not all(p.exists() for p in paths):continue
   rs=[json.loads(p.read_text()) for p in paths];crit=[np.mean([r['summary']['point_rate'] for r in rs]),np.mean([r['summary']['mean_difference'] for r in rs]),-step];options.append((crit,step))
  assert max(options)[1]==c['step'];assert sha(c['path'])==c['sha256']
 decision=json.loads((root/'decision.json').read_text());checks=[]
 for op in ['heuristic','untrained']:
  arr=[]
  for seed,c in selection.items():
   games=json.loads((root/f"seed-{seed}/validation-{c['step']}-{op}.json").read_text())['games'];groups={}
   for g in games:groups.setdefault(g['prefix_id'],[]).append(g['difference'])
   arr.append(np.array([[np.mean([float(d>0)+.5*float(d==0) for d in ds]),np.mean(ds)] for ds in groups.values()]))
  x=np.mean(arr,0);ix=np.random.default_rng(1460000).integers(len(x),size=(4000,len(x)));ci=np.quantile(x[ix].mean(1),[.025,.975],axis=0);ok=bool((np.mean(arr,1)[:,0]>=.5).all() and x[:,0].mean()>=.53 and x[:,1].mean()>0 and ci[0,0]>.5);assert ok==decision['checks'][op]['advance'];assert np.allclose(ci[:,0],decision['checks'][op]['point_ci95']);checks.append(ok)
 assert all(checks)==decision['advance']
 write(root/'audit.json',dict(pass_=True,training_games=sources,terminal_targets=targets,initial_policy_logprob_replays=recomputed,unique_initial_model_hash=next(iter(initial_hashes)),eval_games=evalgames,training_validation_prefix_overlap=0,training_validation_visited_state_overlap=len(train_states&evalstates),pool=gradient_checks,seconds=time.perf_counter()-begin,note='Implementation-side replay and arithmetic, not independent secretary implementation.'))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',default=str(ROOT));audit(p.parse_args().root)
