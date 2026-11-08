"""Finite value-head-only terminal-outcome pilot; no human game ingestion."""
from h_distill import *
import h_distill as common
import h_distill_eval as evaluation
from blokus.ppo_value_leaf import value_inputs
R=Path('artifacts/h-terminal-value');common.R=R;evaluation.R=R

def features(p,s):
 b,st=value_inputs(s,p.channels)
 with torch.no_grad():
  g=p.model.global_net(p.model.conv(torch.from_numpy(b[None]).float()))
 return torch.cat((g,torch.from_numpy(st[None])),dim=-1)[0]

def outcome(scores,turn):
 d=scores[turn]-scores[1-turn];return float((d>0)-(d<0))

def generate(seed):
 setup();idx=np.load(INDEX,mmap_mode='r');forbidden=exclusions();p=PPOAgent(seed,'cpu');a=DeepSearch(p,500);started=time.perf_counter()
 for phase,n,base in [('train',32,46100000),('dev',8,46200000)]:
  path=R/f'{phase}-{seed}.json'
  if path.exists():raise RuntimeError('Refuse overwrite generation')
  games=[]
  for gid in range(n):
   rng=random.Random(base+gid);s=State();moves=[];stats=[]
   while not s.done:
    if len(moves)<6:
     qs=legal_moves(s);m=rng.choice(qs).move if qs else None;st={}
    else:
     m=a.act(s,rng);st=copy.deepcopy(a.stats);st.pop('root_scores',None)
    play(s,m,validate=True);moves.append(raw(m));stats.append(st)
   games.append(dict(id=gid,moves=moves,scores=[s.score(0),s.score(1)],stats=stats))
   print('generation',seed,phase,gid+1,flush=True)
  write(path,dict(seed=seed,phase=phase,games=games,source_sha=sha(f'artifacts/ppo-gen3/seed-{seed}.pt'),elapsed_since_start=time.perf_counter()-started))

def dataset(p,phase,idx,forbidden):
 data=[];labels=[];seen=set()
 row=json.loads((R/f'{phase}-{p.model_seed}.json').read_text())
 for game in row['games']:
  s=State();local=[]
  for ply,m in enumerate(game['moves']):
   h=state_hash(s)
   if ply>=6:
    seen.add(h)
    if h not in forbidden and not contains(idx,h):
     local.append((features(p,s),outcome(game['scores'],s.turn),dict(game=game['id'],ply=ply,turn=s.turn,state_hash=h,target=outcome(game['scores'],s.turn))))
   play(s,Move(*m) if m else None,validate=True)
  assert s.done and [s.score(0),s.score(1)]==game['scores'] and local
  for x,y,label in local:data.append((x,y,1/len(local)));labels.append(label)
 return data,labels,seen

def tensor_data(data):
 return torch.stack([t[0] for t in data]),torch.tensor([t[1] for t in data]),torch.tensor([t[2] for t in data])

def train(seed):
 setup();p=PPOAgent(seed,'cpu');idx=np.load(INDEX,mmap_mode='r');forbidden=exclusions()
 tr,labels,ts=dataset(p,'train',idx,forbidden);dv,dl,ds=dataset(p,'dev',idx,forbidden);assert not ts&ds
 x,y,w=tensor_data(tr);vx,vy,vw=tensor_data(dv);head=copy.deepcopy(p.model.value);opt=torch.optim.Adam(head.parameters(),lr=1e-4);rng=np.random.default_rng(seed+5100)
 with torch.no_grad():old=head(x).squeeze(-1).clone();before=float((((head(vx).squeeze(-1)-vy)**2)*vw).sum()/vw.sum())
 curve=[]
 # Game-uniform weighting; rescale weights to mean 1 for minibatch SGD.
 w=w/w.mean()
 for epoch in range(8):
  order=rng.permutation(len(x))
  for start in range(0,len(order),64):
   ix=order[start:start+64];pred=head(x[ix]).squeeze(-1)
   loss=(w[ix]*((pred-y[ix])**2+.1*(pred-old[ix])**2)).mean()
   assert torch.isfinite(loss);opt.zero_grad();loss.backward();assert all(torch.isfinite(t.grad).all() for t in head.parameters());opt.step()
  with torch.no_grad():mse=float((((head(vx).squeeze(-1)-vy)**2)*vw).sum()/vw.sum())
  curve.append(dict(epoch=epoch+1,dev_mse=mse))
 p.model.value.load_state_dict(head.state_dict());src=torch.load(f'artifacts/ppo-gen3/seed-{seed}.pt',map_location='cpu',weights_only=True)
 changed=[k for k,v in p.model.state_dict().items() if not torch.equal(v,src['model'][k])];assert changed and all(k.startswith('value.') for k in changed)
 torch.save(dict(format='terminal-ppo-inference-v2',channels=4,model=p.model.state_dict(),seed=seed,research_name='H-terminal-value',epochs=8,source_sha=sha(f'artifacts/ppo-gen3/seed-{seed}.pt')),R/f'student-{seed}.pt')
 torch.save(dict(head=head.state_dict(),optimizer=opt.state_dict(),rng=rng.bit_generator.state,epochs=8),R/f'training-state-{seed}.pt')
 write(R/f'learning-{seed}.json',dict(before_dev_mse=before,curve=curve,changed=changed,train_labels=labels,dev_labels=dl,updates=8*((len(x)+63)//64),train_games=32,dev_games=8))
 write(R/f'states-{seed}.json',dict(train=sorted(ts),dev=sorted(ds)))
 print('trained',seed,before,curve[-1],len(labels),flush=True)

def prepare():
 setup();forbidden=exclusions();idx=np.load(INDEX,mmap_mode='r');sets={k:set() for k in ('train','dev')}
 for seed in SEEDS:
  r=json.loads((R/f'states-{seed}.json').read_text())
  for k in sets:sets[k].update(r[k])
 assert not sets['train']&sets['dev'];forbidden|=sets['train']|sets['dev']
 for phase,n,sd in [('validation',12,46300000),('confirmation',32,46400000)]:
  rng=random.Random(sd);bank=[]
  while len(bank)<n:
   s=State();ms=[]
   for _ in range(rng.randint(6,10)):
    qs=legal_moves(s);m=rng.choice(qs).move if qs else None;play(s,m,validate=True);ms.append(raw(m))
   h=state_hash(s)
   if s.done or h in forbidden or contains(idx,h):continue
   forbidden.add(h);bank.append(dict(id=len(bank),moves=ms,state_hash=h))
  write(R/f'{phase}-bank.json',dict(seed=sd,prefixes=bank))
 write(R/'models.json',{str(s):dict(original_sha=sha(f'artifacts/ppo-gen3/seed-{s}.pt'),student_sha=sha(R/f'student-{s}.pt')) for s in SEEDS})

def evaluate():
 setup()
 for seed in SEEDS:evaluation.compare('validation','H',seed)
 a=evaluation.aggregate('validation','H');write(R/'validation-summary.json',a)
 gate=a['point']>=.55 and a['difference']>0 and sum(x['point']>=.5 for x in a['by_seed'].values())>=2 and a['time']['0']['p99_ms']<=750
 write(R/'selection.json',dict(gate=gate,reason='predeclared >=55% positive score, two seeds >=50%, p99<=750ms'))
 if gate:
  for seed in SEEDS:evaluation.compare('confirmation','H',seed)
  c=evaluation.aggregate('confirmation','H');write(R/'confirmation-summary.json',c)
  write(R/'adoption.json',dict(improved=c['point_ci95'][0]>.5 and c['difference_ci95'][0]>0,public_change=False))
 print('SELECTION',gate,flush=True)

def audit():
 setup()
 from audit_h_distill import replay
 counts={};sets={k:set() for k in ('train','dev','validation','confirmation')};idx=np.load(INDEX,mmap_mode='r');policy_checks=0
 for seed in SEEDS:
  p=PPOAgent(seed,'cpu');q=PPOAgent(seed,'cpu',checkpoint=R/f'student-{seed}.pt');lr=json.loads((R/f'learning-{seed}.json').read_text())
  assert all(torch.equal(v,q.model.state_dict()[k]) for k,v in p.model.state_dict().items() if not k.startswith('value.'))
  for phase in ('train','dev'):
   row=json.loads((R/f'{phase}-{seed}.json').read_text());by={(r['game'],r['ply']):r for r in lr[f'{phase}_labels']}
   for g in row['games']:
    assert replay(g['moves'])==g['scores'];counts[phase]=counts.get(phase,0)+1;s=State()
    for ply,m in enumerate(g['moves']):
     if ply>=6:sets[phase].add(state_hash(s))
     if (g['id'],ply) in by:
      label=by[(g['id'],ply)];assert label['state_hash']==state_hash(s) and label['target']==outcome(g['scores'],s.turn) and not contains(idx,state_hash(s))
     if phase=='dev' and g['id']==0 and ply in (6,12,20):
      obs,_=p._observation(s)
      with torch.no_grad():a,_=p.model(*collate([obs],'cpu'));b,_=q.model(*collate([obs],'cpu'))
      assert torch.equal(a,b);policy_checks+=1
     play(s,Move(*m) if m else None,validate=True)
 for phase in ('validation','confirmation'):
  bank=json.loads((R/f'{phase}-bank.json').read_text())['prefixes']
  for seed in SEEDS:
   path=R/f'{phase}-H-{seed}.json'
   if not path.exists():continue
   for g in json.loads(path.read_text())['games']:
    ms=bank[g['prefix']]['moves']+g['moves'];scores=replay(ms);assert scores==g['scores'] and g['difference']==scores[g['seat']]-scores[1-g['seat']];counts[phase]=counts.get(phase,0)+1
    s=state(bank[g['prefix']]['moves']);sets[phase].add(state_hash(s))
    for m in g['moves']:
     play(s,Move(*m) if m else None,validate=True);h=state_hash(s);sets[phase].add(h);assert not contains(idx,h)
  if (R/f'{phase}-summary.json').exists():assert json.loads((R/f'{phase}-summary.json').read_text())==evaluation.aggregate(phase,'H')
 overlaps={f'{a}/{b}':len(sets[a]&sets[b]) for i,a in enumerate(sets) for b in list(sets)[i+1:]};assert not any(overlaps.values())
 write(R/'audit.json',dict(passed=True,games=counts,split_overlaps=overlaps,policy_logit_exact_checks=policy_checks,source_index_sha=sha(INDEX)))
 print('AUDIT PASS',counts,flush=True)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('command',choices=['generate','train','prepare','evaluate','audit']);a.add_argument('--seed',type=int);args=a.parse_args()
 if args.command in ('generate','train'):globals()[args.command](args.seed)
 else:globals()[args.command]()
