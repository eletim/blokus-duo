"""Same-experience fixed-batch benchmark; saved artifacts contain equality verdicts."""
import argparse,copy,hashlib,json,random,time
from pathlib import Path
import numpy as np
import torch
from blokus import ppo_runtime as r,ppo_runtime_reference as ref
from blokus.ppo_policy import TerminalNet,collate
from blokus.ppo_native import observe,tables
from blokus.value_experiment import write
ROOT=Path('artifacts/ppo-four-channel-runtime');CK='artifacts/ppo-gen2/seed-1401.pt'

def tensor_hash(d):
 h=hashlib.sha256()
 for k,v in d.items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()

def records_hash(records):
 h=hashlib.sha256()
 for x in records:
  for a in x['obs']:h.update(a.dtype.str.encode());h.update(str(a.shape).encode());h.update(a.tobytes())
  h.update(json.dumps({k:v for k,v in x.items() if k!='obs'},sort_keys=True).encode())
 return h.hexdigest()

def equal(a,b):
 if torch.is_tensor(a):return torch.equal(a,b)
 if isinstance(a,np.ndarray):return np.array_equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b

def initialize(channels,seed=1740000):
 r.seed_all(1740000);m=TerminalNet(channels).cuda()
 if channels==6:m.load_state_dict(torch.load(CK,map_location='cpu',weights_only=True)['model'])
 past=copy.deepcopy(m);opt=torch.optim.Adam(m.parameters(),lr=.0003);r.seed_all(seed);return m,past,opt

def audit_games(games):
 from blokus.engine import State,Move,legal_moves,is_legal,play
 for g in games:
  state=State();targets={x['ply']:x for x in g['targets']}
  for ply,row in enumerate(g['prefix']+g['moves']):
   if ply in targets:
    x=targets[ply];assert x['state_hash']==r.state_hash(state) and x['player']==state.turn and x['return']==np.sign(g['scores'][state.turn]-g['scores'][1-state.turn])
    if g['opponent']!='current':assert x['player']==g['learner_seat']
    qs=legal_moves(state);m=qs[x['action']].move if qs else None;assert (None if m is None else [m.piece,m.orientation,m.row,m.col])==row
   m=Move(*row) if row else None;assert is_legal(state,m) if m else not legal_moves(state);play(state,m,validate=False)
  assert state.done and g['scores']==[state.score(0),state.score(1)]

def benchmark():
 tables();rows=[];equality=[];saved_games=[]
 for channels in [6,4]:
  # Exclude one entire collection/update from each implementation as warmup.
  for mode in ['reference','native','resident']:
   m,p,o=initialize(channels);rec,_,_=ref.collect(m,p,0,1740099) if mode=='reference' else r.collect(m,p,0,1740099,backend='native');ref.update(m,o,rec) if mode=='reference' else r.update(m,o,rec,packed=mode=='resident')
  for repeat in range(3):
   outputs={}
   modes=['reference','native','resident'] if repeat%2==0 else ['resident','native','reference']
   for mode in modes:
    m,p,o=initialize(channels,1740100+repeat);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
    rec,games,gen=ref.collect(m,p,repeat,1740100+repeat) if mode=='reference' else r.collect(m,p,repeat,1740100+repeat,backend='native')
    up=ref.update(m,o,rec) if mode=='reference' else r.update(m,o,rec,packed=mode=='resident');torch.cuda.synchronize();wall=time.perf_counter()-start
    saving=r.save('/tmp/ppo-runtime-bench.pt',m,p,o,repeat+1,1740100+repeat,'native',mode=='resident',[])
    outputs[mode]=dict(model=copy.deepcopy(m.state_dict()),optimizer=copy.deepcopy(o.state_dict()),rng=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),python=random.getstate(),numpy=np.random.get_state(),games=games,inputs=records_hash(rec))
    row=dict(channels=channels,mode=mode,repeat=repeat,games=16,learner_states=len(rec),optimizer_steps=up['updates'],candidates=sum(len(x['obs'][2]) for x in rec),max_candidates=max(len(x['obs'][2]) for x in rec),wall=wall,save_seconds=saving,generation=gen,update=up,peak_allocated_bytes=torch.cuda.max_memory_allocated(),games_per_second=16/wall,learner_states_per_second=len(rec)/wall,optimizer_steps_per_second=up['updates']/wall,model_hash=tensor_hash(m.state_dict()),input_hash=outputs[mode]['inputs']);rows.append(row);print(channels,mode,repeat,round(wall,4),row['model_hash'][:12],flush=True)
   for mode in ['native','resident']:
    match=equal(outputs['reference'],outputs[mode]);equality.append(dict(channels=channels,repeat=repeat,mode=mode,bit_equal=match));assert match,('not bit equal',channels,repeat,mode)
   saved_games.extend(outputs['reference']['games'])
 # Audits occur after all timed measurements.
 audit_games(saved_games);write(ROOT/'benchmark.json',dict(rows=rows,equality=equality,all_bit_equal=True,game_audit=True,measured_games=len(rows)*16,warmup_games=96,warmup_updates=6,initialization_excluded=True,compile=False));write(ROOT/'benchmark-games.json',saved_games)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['benchmark']);p.parse_args();benchmark()
