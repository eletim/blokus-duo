"""Paired-seat extraction control; unchanged original teacher/training algorithm."""
from h_distill import *
import h_distill as base
R=Path('artifacts/h-distill-balanced');base.R=R
STAGES=(12,20,28)
def select(rows,quota):
 assert quota>0 and quota%2==0
 arms={'A':[],'B':[]};selected=[];counts={}
 for stage in STAGES:
  pairs=[]
  for gid in sorted({r['game'] for r in rows}):
   pair=[r for r in rows if r['game']==gid and r['ply'] in (stage,stage+1)]
   if len(pair)!=2 or {r['turn'] for r in pair}!={0,1} or any(r['legal_count']<2 or r['stats']['depth']!=6 for r in pair):continue
   pairs.append(sorted(pair,key=lambda r:r['turn']))
  assert len(pairs)>=quota,(stage,len(pairs),quota)
  counts[str(stage)]=dict(eligible_pairs=len(pairs),selected_pairs=quota)
  for i,pair in enumerate(pairs[:quota]):
   arms['A'].append(pair[0]);arms['B'].append(pair[i%2]);selected.extend(pair)
 for name,rs in arms.items():
  counts[name]={f'{st}/{turn}':sum(r['ply']//2==st//2 and r['turn']==turn for r in rs) for st in STAGES for turn in (0,1)}
  assert len(rs)==3*quota
  for st in STAGES:
   assert counts[name][f'{st}/0']==(quota if name=='A' else quota//2)
   assert counts[name][f'{st}/1']==(0 if name=='A' else quota//2)
 return arms,selected,counts

def generate(phase,n):
 idx=np.load(INDEX,mmap_mode='r');forbidden=base.exclusions();started=time.perf_counter()
 external={'pilot':39100100,'train':39200100,'dev':39300100}[phase]
 for seed in SEEDS:
  out=R/f'{phase}-{seed}.json'
  if out.exists():raise RuntimeError('preserve existing output; do not silently rerun')
  p=PPOAgent(seed,'cpu',backend='python');teacher=DeepSearch(p,60000,width=4,max_depth=6);games=[];labels=[];dropped=[]
  for gid in range(n):
   rng=random.Random(external+gid);s=State();moves=[];visited=[]
   while not s.done:
    ply=len(moves);h=state_hash(s)
    if ply>=6 and h in forbidden:raise RuntimeError('old evaluation state encountered')
    visited.append(h);obs,qs=p._observation(s)
    if ply in (12,13,20,21,28,29):
     if len(qs)<2 or contains(idx,h):dropped.append(dict(game=gid,ply=ply,legal_count=len(qs),reason='singleton/pass or training overlap'))
     else:
      greedy=p.act(s,random.Random(0));m=teacher.act(s,random.Random(0));stats=copy.deepcopy(teacher.stats)
      with torch.no_grad():l,_=p.model(*collate([obs],'cpu'));order=l[0].argsort(descending=True).tolist()[:4]
      roots=[qs[i].move for i in order]
      if greedy not in roots:roots[-1]=greedy
      if stats['depth']==6:
       scores=stats['root_scores'];targets=[raw(roots[i]) for i,v in enumerate(scores) if abs(v-max(scores))<=1e-7]
       labels.append(dict(game=gid,ply=ply,turn=s.turn,stage=ply-ply%2,legal_count=len(qs),state_hash=h,moves=copy.deepcopy(moves),target=raw(m),targets=targets,greedy=raw(greedy),changed=m!=greedy,stats=stats))
      else:dropped.append(dict(game=gid,ply=ply,reason='incomplete teacher'))
    if not qs:m=None
    elif ply<6:m=rng.choice(qs).move
    elif rng.random()<.2:
     with torch.no_grad():l,_=p.model(*collate([obs],'cpu'));ix=l[0].argsort(descending=True).tolist()[:8]
     m=qs[rng.choice(ix)].move
    else:m=p.act(s,rng)
    play(s,m,validate=True);moves.append(raw(m))
   games.append(dict(id=gid,moves=moves,visited=visited+[state_hash(s)],scores=[s.score(0),s.score(1)]))
   if gid%16==0:print(phase,seed,gid,len(labels),flush=True)
  write(out,dict(seed=seed,phase=phase,external_seed=external,games=games,labels=labels,dropped=dropped,teacher_model_sha256=sha(f'artifacts/ppo-gen3/seed-{seed}.pt')))
 write(R/f'{phase}-summary.json',dict(wall_seconds=time.perf_counter()-started,games_per_seed=n))
def prepare_data():
 report={}
 for seed in SEEDS:
  train=json.loads((R/f'train-{seed}.json').read_text());dev=json.loads((R/f'dev-{seed}.json').read_text())
  arms,_,counts=select(train['labels'],64);_,devrows,dc=select(dev['labels'],16)
  for arm,rows in arms.items():write(R/f'{arm}-train-{seed}.json',dict(labels=rows,source_sha=sha(R/f'train-{seed}.json')))
  write(R/f'balanced-dev-{seed}.json',dict(labels=devrows,source_sha=sha(R/f'dev-{seed}.json')))
  report[str(seed)]=dict(train=counts,dev=dc,labels_per_arm=192,dev_labels=96,unique_games={arm:len({r['game'] for r in rs}) for arm,rs in arms.items()},paired_games_identical=[(r['game'],r['stage']) for r in arms['A']]==[(r['game'],r['stage']) for r in arms['B']])
  assert report[str(seed)]['paired_games_identical']
 write(R/'pre-training-counts.json',report)
if __name__=='__main__':
 base.setup();a=argparse.ArgumentParser();a.add_argument('phase',choices=['pilot','train','dev','prepare']);a.add_argument('--games',type=int,default=4);args=a.parse_args()
 if args.phase=='prepare':prepare_data()
 else:generate(args.phase,args.games)
