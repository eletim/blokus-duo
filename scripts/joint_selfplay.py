"""Finite on-policy joint PPO continuation against a frozen opponent pool."""
from h_distill import *
import h_distill as common
import h_distill_eval as evaluation
import gzip,subprocess,os
from blokus import ppo_runtime as runtime
from blokus.ppo_gpu import update_fast
R=Path('artifacts/joint-selfplay');common.R=R;evaluation.R=R
CACHE=Path('/tmp/blokus-joint-selfplay-training.npy')

def base_checkpoint(seed):return Path(f'/home/eletim/blokus-duo-ai/artifacts/ppo-lr64/B-seed-{seed}/iteration-5120.pt')
def modes():return ['current','past','anchor','heuristic']*4

def train(seed,until,name='train',resume=None):
 setup();out=R/f'{name}-{seed}';out.mkdir(parents=True,exist_ok=True)
 anchor=PPOAgent(seed,'cuda').model
 if resume:
  model,past,opt,ck=runtime.restore(resume,'cuda');assert ck['seed']==seed and ck['joint_name']==name;step=ck['iteration'];logs=ck['logs'];history=ck['history_segments'];assert all(x['lr']==1e-4 for x in opt.param_groups)
 else:
  if (out/'initial.pt').exists():raise RuntimeError('Refuse overwrite; resume explicitly')
  model,past,opt,ck=runtime.restore(base_checkpoint(seed),'cuda');assert all(torch.equal(v,anchor.state_dict()[k]) for k,v in model.state_dict().items());past.load_state_dict(model.state_dict())
  for g in opt.param_groups:g['lr']=1e-4
  runtime.seed_all(seed+61000+(1000 if name.startswith('pilot') else 0));step=0;logs=[];history=[]
  runtime.save(out/'initial.pt',model,past,opt,0,seed,'native',True,[])
 forbidden=exclusions();streamseed=seed+62000+(1000 if name.startswith('pilot') else 0);start=time.perf_counter();initial=step
 while step<until:
  end=min(step+256,until);first=step+1;path=out/f'history-{first}-{end}.jsonl.gz'
  if path.exists():raise RuntimeError(f'History exists: {path}')
  with gzip.open(path,'wt',compresslevel=1) as f:
   while step<end:
    rec,games,tally=runtime.collect(model,past,step,streamseed,backend='native',forbidden_starts=forbidden,fixed_pool={'anchor':anchor},opponent_modes=modes())
    metric=update_fast(model,opt,rec,packed=True,minibatch=64);step+=1
    if step%4==0:past.load_state_dict(model.state_dict())
    f.write(json.dumps(dict(iteration=step,games=games),separators=(',',':'))+'\n')
    logs.append(dict(iteration=step,generation=tally,update=metric,games_sha=hashlib.sha256(json.dumps(games,sort_keys=True).encode()).hexdigest()))
    if len(logs)>=64 and np.mean([x['update']['entropy'] for x in logs[-64:]])<.05:raise RuntimeError('entropy collapse')
    if len(logs)>=32 and np.mean([x['update']['approx_kl'] for x in logs[-32:]])>.2:raise RuntimeError('KL collapse')
    if step%64==0:print(seed,step,'seconds',round(time.perf_counter()-start,2),flush=True)
  history.append(dict(path=path.name,sha=sha(path),first=first,last=step))
  dest=out/f'iteration-{step}.pt';runtime.save(dest,model,past,opt,step,seed,'native',True,logs)
  ck=torch.load(dest,map_location='cpu',weights_only=False);ck.update(joint_name=name,history_segments=history,base_sha=sha(base_checkpoint(seed)),original_export_sha=sha(f'artifacts/ppo-gen3/seed-{seed}.pt'),stream_seed=streamseed,learning_rate=1e-4,minibatch=64,opponent_modes=modes());torch.save(ck,dest)
  runtime.export(dest,out/f'inference-{step}.pt')
 result=dict(seed=seed,iteration=step,games=(step-initial)*16,wall_seconds=time.perf_counter()-start,history_segments=history,checkpoint=str(dest));write(out/f'result-{step}.json',result);return result

def run_training():
 setup();plan=json.loads((R/'training-size.json').read_text());n=plan['iterations'];results=[];start=time.perf_counter()
 for seed in SEEDS:
  results.append(train(seed,n));subprocess.run([sys.executable,'-c',f'import shutil;shutil.copyfile("{R}/train-{seed}/inference-{n}.pt","{R}/student-{seed}.pt")'],check=True)
 write(R/'training-complete.json',dict(iterations=n,seeds=SEEDS,games_per_seed=n*16,results=results,wall_seconds=time.perf_counter()-start))

def audit_training():
 setup()
 from blokus.engine import is_legal,PLACEMENTS
 from blokus.ppo_native import scan
 from audit_h_distill import replay
 new=set();counts={};sampled=0;targets_count=0;start=time.perf_counter();complete=json.loads((R/'training-complete.json').read_text());n=complete['iterations'];forbidden=exclusions();old_hits=set()
 for name,end in [('pilot',8),('train',n)]:
  for seed in (SEEDS[:1] if name=='pilot' else SEEDS):
   out=R/f'{name}-{seed}';ck=torch.load(out/f'iteration-{end}.pt',weights_only=False,map_location='cpu');assert ck['base_sha']==sha(base_checkpoint(seed)) and ck['original_export_sha']==sha(f'artifacts/ppo-gen3/seed-{seed}.pt');ng=0
   for seg in ck['history_segments']:
    assert sha(out/seg['path'])==seg['sha']
    with gzip.open(out/seg['path'],'rt') as f:
     for line in f:
      row=json.loads(line);it=row['iteration'];assert hashlib.sha256(json.dumps(row['games'],sort_keys=True).encode()).hexdigest()==ck['logs'][it-1]['games_sha']
      assert np.isfinite(list(ck['logs'][it-1]['update'].values())).all()
      for slot,g in enumerate(row['games']):
       assert g['opponent']==modes()[slot] and g['learner_seat']==(slot//4)%2
       s=State();ts={x['ply']:x for x in g['targets']};assert len(ts)==len(g['targets'])
       for ply,m in enumerate(g['prefix']+g['moves']):
        h=state_hash(s);new.add(bytes.fromhex(h));move=Move(*m) if m else None
        assert is_legal(s,move) if move else len(scan(s,False)[0])==0
        if ply>=len(g['prefix']):
         assert (ply in ts)==(g['opponent']=='current' or s.turn==g['learner_seat'])
         if h in forbidden:old_hits.add(h)
        if ply in ts:
         t=ts[ply];assert t['state_hash']==h and t['player']==s.turn and t['return']==float(np.sign(g['scores'][s.turn]-g['scores'][1-s.turn])) and np.isfinite([t['log_prob'],t['value']]).all()
         if targets_count%256==0:
          qs=legal_moves(s);assert (qs[t['action']].move if qs else None)==move
         targets_count+=1
        play(s,move,validate=False)
       new.add(bytes.fromhex(state_hash(s)));assert s.done and g['scores']==[s.score(0),s.score(1)]
       if ng%256==0:assert replay(g['prefix']+g['moves'])==g['scores'];sampled+=1
       ng+=1
   assert ng==end*16;counts[f'{name}-{seed}']=ng;print('training audited',name,seed,ng,flush=True)
 a=np.array(sorted(new),dtype='S32');np.save(CACHE,a)
 write(R/'training-audit.json',dict(passed=True,games=counts,all_games_legality_labels_scores_checked=True,independent_coordinate_sample_games=sampled,action_index_sample_stride=256,targets=targets_count,visited_states=len(a),index_sha=sha(CACHE),incidental_old_evaluation_start_hits=len(old_hits),wall_seconds=time.perf_counter()-start))

def prepare():
 setup();idx=np.load(INDEX,mmap_mode='r');new=np.load(CACHE,mmap_mode='r');forbidden=exclusions()
 for phase,n,sd in [('validation',12,56100100),('confirmation',32,56200100)]:
  rng=random.Random(sd);bank=[]
  while len(bank)<n:
   s=State();ms=[]
   for _ in range(rng.randint(6,10)):
    qs=legal_moves(s);m=rng.choice(qs).move if qs else None;play(s,m,validate=True);ms.append(raw(m))
   h=state_hash(s)
   if s.done or h in forbidden or contains(idx,h) or contains(new,h):continue
   forbidden.add(h);bank.append(dict(id=len(bank),moves=ms,state_hash=h))
  write(R/f'{phase}-bank.json',dict(seed=sd,prefixes=bank))
 write(R/'models.json',{str(s):dict(original_sha=sha(f'artifacts/ppo-gen3/seed-{s}.pt'),student_sha=sha(R/f'student-{s}.pt')) for s in SEEDS})

def evaluate():
 setup()
 for seed in SEEDS:evaluation.compare('validation','H',seed,backend='native')
 a=evaluation.aggregate('validation','H');write(R/'validation-summary.json',a)
 gate=a['point']>=.55 and a['difference']>0 and sum(x['point']>=.5 for x in a['by_seed'].values())>=2 and a['time']['0']['p99_ms']<=750
 write(R/'selection.json',dict(gate=gate))
 if gate:
  for seed in SEEDS:evaluation.compare('confirmation','H',seed,backend='native')
  c=evaluation.aggregate('confirmation','H');write(R/'confirmation-summary.json',c);write(R/'adoption.json',dict(improved=c['point_ci95'][0]>.5 and c['difference_ci95'][0]>0 and c['by_seed']['1751401']['point']>.5 and c['by_seed']['1751401']['difference']>0,public_change=False))
 print('SELECTION',gate,flush=True)

def audit_evaluation():
 setup();new=np.load(CACHE,mmap_mode='r');old=np.load(INDEX,mmap_mode='r');seen={};counts={}
 from audit_h_distill import replay
 for phase in ('validation','confirmation'):
  bank=json.loads((R/f'{phase}-bank.json').read_text())['prefixes'];seen[phase]=set();n=0
  for seed in SEEDS:
   path=R/f'{phase}-H-{seed}.json'
   if not path.exists():continue
   for g in json.loads(path.read_text())['games']:
    ms=bank[g['prefix']]['moves']+g['moves'];assert replay(ms)==g['scores'];assert g['difference']==g['scores'][g['seat']]-g['scores'][1-g['seat']];s=state(bank[g['prefix']]['moves'])
    for m in g['moves']+[None]:
     h=state_hash(s);assert not contains(new,h) and not contains(old,h);seen[phase].add(h)
     if s.done:break
     play(s,Move(*m) if m else None,validate=True)
    n+=1
  counts[phase]=n
  if n:assert json.loads((R/f'{phase}-summary.json').read_text())==evaluation.aggregate(phase,'H')
 assert not seen['validation']&seen['confirmation'];write(R/'evaluation-audit.json',dict(passed=True,games=counts,visited_state_overlap=0))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['train','run_training','audit_training','prepare','evaluate','audit_evaluation']);p.add_argument('--seed',type=int,default=1751401);p.add_argument('--until',type=int,default=8);p.add_argument('--name',default='pilot');p.add_argument('--resume');a=p.parse_args()
 if a.command=='train':train(a.seed,a.until,a.name,a.resume)
 else:globals()[a.command]()
