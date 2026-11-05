"""Same states and models, alternating backend microbench and PPO equivalence."""
import argparse,contextlib,copy,json,random,time,hashlib
from pathlib import Path
import numpy as np
import torch
from blokus import ai,root_policy,terminal_ppo as p,cpu_reference as ref
from blokus.engine import State,legal_moves,play,Move
from blokus.research import clone,sha
from blokus.value_search import encode
from blokus.value_experiment import write
R=Path('artifacts/ppo-cpu-throughput');MODEL='artifacts/ppo-gen2/seed-1401.pt'
@contextlib.contextmanager
def backend(old):
 targets=[(ai,'features'),(root_policy,'features'),(p,'features'),(root_policy,'inputs'),(p,'inputs'),(p,'observe')];saved=[getattr(m,k) for m,k in targets]
 if old:
  for m,k in targets:setattr(m,k,getattr(ref,k))
 try:yield
 finally:
  for (m,k),v in zip(targets,saved):setattr(m,k,v)

def dataset():
 path=R/'games.json'
 if path.exists():games=json.loads(path.read_text())
 else:
  agent=p.TerminalAgent(MODEL,'cpu');games=[]
  for seed in range(1730000,1730012):
   rng=random.Random(seed);s=State();moves=[]
   while not s.done:
    qs=legal_moves(s)
    if s.turn==0:m=agent.act(s,rng)
    elif seed%2:m=rng.choice(qs).move if qs else None
    else:m=qs[int(np.argmax(ai.heuristic(ai.features(s,qs)[0])))].move if qs else None
    moves.append(None if m is None else [m.piece,m.orientation,m.row,m.col]);play(s,m,validate=False)
   games.append(dict(seed=seed,moves=moves,scores=[s.score(0),s.score(1)]))
  write(path,games)
 states=[]
 for game in games:
  s=State()
  for ply,row in enumerate(game['moves']):
   if ply in [0,4,12,24,36] or row is None:states.append(clone(s))
   play(s,Move(*row) if row else None,validate=False)
 return states

def micro(label):
 states=dataset();qs=[legal_moves(s) for s in states];rows=[]
 agents={d:p.TerminalAgent(MODEL,d) for d in ['cpu','cuda']}
 for repeat in range(4):
  for old in ([True,False] if repeat%2==0 else [False,True]):
   with backend(old):
    operations={'legal':lambda:[legal_moves(s) for s in states], 'features12':lambda:[ai.features(s,x) for s,x in zip(states,qs)],'inputs39':lambda:[root_policy.inputs(s,x) for s,x in zip(states,qs)],'encode':lambda:[encode(s) for s in states], 'observe_with_legal':lambda:[p.observe(s) for s in states]}
    for device,agent in agents.items():operations['inference_'+device]=lambda a=agent:[a.act(s,random.Random(1)) for s in states]
    for name,fn in operations.items():
     fn();torch.cuda.synchronize();t=time.perf_counter()
     for _ in range(3):fn()
     torch.cuda.synchronize();rows.append(dict(repeat=repeat,backend='old' if old else 'new',operation=name,ms_per_state=(time.perf_counter()-t)*1000/(3*len(states))))
 write(R/f'micro-{label}.json',dict(states=len(states),games_sha256=sha(R/'games.json'),model_sha256=sha(MODEL),rows=rows,warmup='one whole bank per operation, excluded; initialization excluded'))
 print(label,{name:{b:np.median([r['ms_per_state'] for r in rows if r['operation']==name and r['backend']==b]) for b in ['old','new']} for name in operations},flush=True)

def digest_records(records):
 h=hashlib.sha256()
 for r in records:
  for a in r['obs']:h.update(a.dtype.str.encode());h.update(str(a.shape).encode());h.update(a.tobytes())
  h.update(json.dumps({k:v for k,v in r.items() if k!='obs'},sort_keys=True).encode())
 return h.hexdigest()

def equal(a,b):
 if torch.is_tensor(a):return torch.equal(a,b)
 if isinstance(a,np.ndarray):return np.array_equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b

def training():
 rows=[];reference={}
 # Kernel warmup before measurements, without advancing any measured RNG stream.
 a=p.TerminalAgent(MODEL);warm,_,_=p.collect(a.model,copy.deepcopy(a.model),0,1731000,16);p.update(a.model,torch.optim.Adam(a.model.parameters(),lr=.0003),warm)
 for repeat in range(3):
  pair={}
  for old in ([True,False] if repeat%2==0 else [False,True]):
   ai.seed_all(1732000+repeat);model=p.TerminalNet().cuda();model.load_state_dict(torch.load(MODEL,weights_only=False)['model']);past=copy.deepcopy(model);opt=torch.optim.Adam(model.parameters(),lr=.0003);ai.seed_all(1732000+repeat);games=[];inputs=[];logs=[];torch.cuda.synchronize();start=time.perf_counter()
   with backend(old):
    for iteration in [0,1]:
     rec,gs,gen=p.collect(model,past,iteration,1732000+repeat,16);inputs.append(digest_records(rec));games.extend(gs);up=p.update(model,opt,rec);logs.append(dict(generation=gen,update=up))
     if (iteration+1)%4==0:past.load_state_dict(model.state_dict())
   torch.cuda.synchronize();wall=time.perf_counter()-start
   state=dict(model=copy.deepcopy(model.state_dict()),past=copy.deepcopy(past.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),inputs=inputs,games=games)
   pair[old]=state;generation=sum(x['generation']['seconds'] for x in logs);row=dict(repeat=repeat,backend='old' if old else 'new',games=32,iterations=2,wall=wall,generation_seconds=generation,games_per_second=32/generation,logs=logs,inputs=inputs,games_sha256=hashlib.sha256(json.dumps(games,sort_keys=True).encode()).hexdigest());rows.append(row);print('training',row['backend'],repeat,wall,flush=True)
  assert equal(pair[True],pair[False]),'trajectory/input/model/optimizer/RNG mismatch'
  from blokus.engine import is_legal
  for g in pair[False]['games']:
   state=State();targets={x['ply']:x for x in g['targets']}
   for ply,row in enumerate(g['prefix']+g['moves']):
    if ply in targets:
     target=targets[ply];assert target['player']==state.turn and target['return']==float(np.sign(g['scores'][state.turn]-g['scores'][1-state.turn]))
     qs=legal_moves(state);action=qs[target['action']].move if qs else None;assert (None if action is None else [action.piece,action.orientation,action.row,action.col])==row
    move=Move(*row) if row else None;assert is_legal(state,move) if move else not legal_moves(state);play(state,move,validate=False)
   assert state.done and g['scores']==[state.score(0),state.score(1)]
 write(R/'training.json',dict(pass_=True,bit_equal=['trajectory','all observation arrays','model','past','optimizer','Python NumPy Torch CUDA RNG'],rows=rows,additional_training_games=192,production_training=False,reward_action_legal_replay_pass=True,warmup='one 16-game collect and update excluded; all measured models/optimizer/RNG reinitialized from same checkpoint/seed'))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['micro','training']);ap.add_argument('--label',default='before');args=ap.parse_args();ai.seed_all(1730000);R.mkdir(exist_ok=True)
 micro(args.label) if args.mode=='micro' else training()
