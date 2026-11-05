"""Equal-depth student/teacher/zero/random union on stratified heldout roots."""
import sys,json,random,hashlib
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from blokus.scaling_data import ROOT,load
from blokus.correction import CorrectionNet
from blokus.value_search import ValueEvaluator,encode
from blokus.aligned_data import BASE
from blokus.deep_search import TreeSearch
from blokus.engine import State,Move,legal_moves
from blokus.node_diagnostic import child
from blokus.root_policy import inputs
from blokus.ai import seed_all
from blokus.value_experiment import write
from blokus.research import sha

def main():
 seed_all(1100);size=int(sys.argv[1]);models={}
 for c in ['plain','decay']:
  m=CorrectionNet().cuda();m.load_state_dict(torch.load(ROOT/f'n{size}-f0-s1101-{c}/step-1000.pt',map_location='cpu',weights_only=False)['model']);m.eval();models[c]=m
 # 3 pools x2 seats x2 phases x3 games; fixed fold0, unique roots.
 records=load('teacher-labels');groups={};selected=[]
 for r in records:
  if not r['bank'] or r['fold']!=0:continue
  key=(r['pool'],r['seat'],r['phase']);groups.setdefault(key,[])
  if len(groups[key])<3:groups[key].append(r);selected.append(r)
 tree=TreeSearch(ValueEvaluator(BASE,'cuda',.5,24),width=3,tt_limit=0);out=[]
 for r in selected:
  s=State(**r['state']);by={q.move:q for q in legal_moves(s)};qs=[by[Move(*m)] for m in r['moves']];x,_=inputs(s,qs);b,st=encode(s);initial=np.array(r['initial']);known=[i for i,v in enumerate(r['final']) if v is not None]
  picks=dict(zero=int(initial.argmax()),teacher=max(known,key=lambda i:r['final'][i]))
  with torch.no_grad():
   for c,m in models.items():
    pred=m(torch.as_tensor(b[None],device='cuda'),torch.as_tensor(st[None],device='cuda'),torch.as_tensor(x[None],device='cuda'))[0].cpu().numpy();picks[c]=int((initial+.5*pred).argmax())
  rng=random.Random(1140000+r['game']+r['ply']);rand=rng.sample(range(len(qs)),min(2,len(qs)));ids=sorted(set(picks.values())|set(rand));truth={str(i):-tree.search(child(s,qs[i].move),3) for i in ids};best=max(truth.values())
  out.append(dict(game=r['game'],state_hash=r['state_hash'],picks=picks,random=rand,depth=4,truth=truth,regret={c:best-truth[str(i)] for c,i in picks.items()},student_originally_labeled={c:picks[c] in known for c in models}))
 summary={c:float(np.mean([r['regret'][c] for r in out])) for c in ['zero','teacher','plain','decay']}
 write(ROOT/'bias-diagnostic.json',dict(size=size,checkpoints={c:dict(path=str(ROOT/f'n{size}-f0-s1101-{c}/step-1000.pt'),sha256=sha(ROOT/f'n{size}-f0-s1101-{c}/step-1000.pt')) for c in models},rows=out,unique_games=len({r['game'] for r in out}),roots=len(out),mean_regret=summary,note='Fixed seed1101 fold0, post-selection diagnostic only. Teacher is best among original completed labels; union reference is selective depth4, not true game value. Unknown student/random choices completed at same depth.'))
 print(summary,flush=True)
if __name__=='__main__':main()
