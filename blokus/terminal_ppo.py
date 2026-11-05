"""All-legal spatial policy/value, terminal win/loss PPO; no search teacher."""
import argparse,copy,json,os,random,time,gzip
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .research import clone,source_hashes,sha
from .ai import features,heuristic,seed_all
from .root_policy import inputs
from .value_search import encode
from .engine import State,legal_moves,play,Move
from .value_experiment import write
from .visited_data import state_hash
ROOT=Path('artifacts/terminal-ppo')

def terminal_return(s,player):
 if not s.done:raise ValueError('incomplete game is not a terminal target')
 return float(np.sign(s.score(player)-s.score(1-player)))

def observe(s,qs=None):
 if s.done:raise ValueError('terminal has no policy actions')
 qs=legal_moves(s) if qs is None else qs;b,st=encode(s)
 if qs:
  x,_=inputs(s,qs);cells=np.full((len(qs),5),196,np.int64)
  for i,q in enumerate(qs):
   for j,(y,col) in enumerate(q.move.cells):cells[i,j]=14*y+col
 else:x=np.zeros((1,39),np.float32);cells=np.full((1,5),196,np.int64)
 return (b,st,x,cells),qs

def collate(obs,device='cuda'):
 n=max(len(o[2]) for o in obs);b=len(obs);x=np.zeros((b,n,39),np.float32);cells=np.full((b,n,5),196,np.int64);mask=np.zeros((b,n),bool)
 for i,o in enumerate(obs):k=len(o[2]);x[i,:k]=o[2];cells[i,:k]=o[3];mask[i,:k]=True
 return tuple(torch.as_tensor(a,device=device) for a in (np.stack([o[0] for o in obs]),np.stack([o[1] for o in obs]),x,cells,mask))

class TerminalNet(nn.Module):
 def __init__(self):
  super().__init__();self.conv=nn.Sequential(nn.Conv2d(6,8,3,padding=1),nn.ReLU(),nn.Conv2d(8,8,3,padding=1),nn.ReLU());self.global_net=nn.Sequential(nn.Flatten(),nn.Linear(8*196,32),nn.Tanh());self.position=nn.Embedding(197,16,padding_idx=196)
  self.policy=nn.Sequential(nn.Linear(32+49+39+16+8,64),nn.Tanh(),nn.Linear(64,1));self.value=nn.Sequential(nn.Linear(32+49,64),nn.Tanh(),nn.Linear(64,1),nn.Tanh())
  nn.init.zeros_(self.policy[-1].weight);nn.init.zeros_(self.policy[-1].bias);nn.init.zeros_(self.value[-2].weight);nn.init.zeros_(self.value[-2].bias)
  self.register_buffer('prior',torch.tensor([12,4,2,1.2,2,.3,1,.2,-.6,0,0,.2],dtype=torch.float32))
 def forward(self,board,state,x,cells,mask):
  f=self.conv(board.float());g=self.global_net(f);b,n=x.shape[:2];count=(cells!=196).sum(-1).clamp_min(1).unsqueeze(-1)
  pos=self.position(cells).sum(-2)/count
  flat=nn.functional.pad(f.flatten(2),(0,1));idx=cells.reshape(b,1,-1).expand(-1,8,-1);local=flat.gather(2,idx).reshape(b,8,n,5).sum(-1).transpose(1,2)/count
  state_g=torch.cat((g,state),-1);parts=torch.cat((state_g[:,None].expand(-1,n,-1),x,pos,local),-1)
  logits=x[...,:12]@self.prior+self.policy(parts).squeeze(-1);return logits.masked_fill(~mask,-float('inf')),self.value(state_g).squeeze(-1)

def sample_batch(model,obs):
 with torch.no_grad():
  logits,values=model(*collate(obs));dist=torch.distributions.Categorical(logits=logits);actions=dist.sample();lp=dist.log_prob(actions)
 return actions.cpu().tolist(),lp.cpu().tolist(),values.cpu().tolist()

def pool_modes(iteration,batch,kind="A"):
 assert kind in ("A","B") and batch%8==0
 modes=["current","current","past","heuristic"]*(batch//4)
 if kind=="B":
  for j,i in enumerate(range(2,batch,4)):modes[i]=("past","gen2","gen2048")[(iteration+j)%3]
 return modes

def collect(model,past,iteration,seed,batch=16,forbidden_starts=None,fixed_pool=None):
 assert batch%8==0
 states=[State() for _ in range(batch)];hist=[[] for _ in states];traces=[[] for _ in states];records=[];modes=pool_modes(iteration,batch,'B' if fixed_pool else 'A');seats=[(i//4)%2 for i in range(batch)];tally=dict(cpu_generation_pack=0.,gpu_inference_sync=0.,other=0.);start=time.perf_counter();rngs=[];prefixes=[];prefix_rejections=[]
 for i,s in enumerate(states):
  rng=random.Random(seed*100000+(iteration*batch+i));rngs.append(rng);rejected=0
  while True:
   prefix=[];blocked=False
   for _ in range(rng.randint(2,6)):
    qs=legal_moves(s);m=rng.choice(qs).move if qs else None;play(s,m,validate=False);prefix.append(None if m is None else [m.piece,m.orientation,m.row,m.col])
    if forbidden_starts and state_hash(s) in forbidden_starts:blocked=True
   if not blocked:break
   rejected+=1
   if rejected>100:raise RuntimeError('prefix rejection pool exhausted')
   s=State();states[i]=s
  prefix_rejections.append(rejected);prefixes.append(prefix)
 while not all(s.done for s in states):
  models={'current':model,'past':past,**(fixed_pool or {})};groups={k:[] for k in models};prepared={}
  for i,s in enumerate(states):
   if s.done:continue
   t=time.perf_counter();qs=legal_moves(s);mode='current' if modes[i]=='current' or s.turn==seats[i] else modes[i]
   if mode=='heuristic':
    m=qs[int(np.argmax(heuristic(features(s,qs)[0])))].move if qs else None;hist[i].append(None if m is None else [m.piece,m.orientation,m.row,m.col]);play(s,m,validate=False)
   else:
    obs,_=observe(s,qs);prepared[i]=(obs,qs,s.turn,state_hash(s));groups[mode].append(i)
   tally['cpu_generation_pack']+=time.perf_counter()-t
  for mode,ids in groups.items():
   if not ids:continue
   t=time.perf_counter();actions,lp,vs=sample_batch(models[mode],[prepared[i][0] for i in ids]);tally['gpu_inference_sync']+=time.perf_counter()-t
   for j,i in enumerate(ids):
    obs,qs,player,h=prepared[i];a=actions[j];m=qs[a].move if qs else None
    if mode=='current':traces[i].append(dict(obs=obs,action=a,log_prob=lp[j],value=vs[j],player=player,state_hash=h,ply=len(prefixes[i])+len(hist[i])))
    hist[i].append(None if m is None else [m.piece,m.orientation,m.row,m.col]);play(states[i],m,validate=False)
 games=[]
 for i,s in enumerate(states):
  for r in traces[i]:r['return']=terminal_return(s,r['player']);r['opponent']=modes[i];records.append(r)
  games.append(dict(id=seed*100000+iteration*batch+i,opponent=modes[i],learner_seat=seats[i],prefix=prefixes[i],moves=hist[i],scores=[s.score(0),s.score(1)],targets=[{k:r[k] for k in ['ply','player','return','state_hash','action','log_prob','value']} for r in traces[i]]))
 if forbidden_starts is not None:
  for g,rejected in zip(games,prefix_rejections):g['prefix_rejections']=rejected
 tally['seconds']=time.perf_counter()-start;tally['other']=tally['seconds']-tally['cpu_generation_pack']-tally['gpu_inference_sync'];return records,games,tally

def update(model,opt,records,epochs=2):
 start=time.perf_counter();ret=torch.tensor([r['return'] for r in records],device='cuda');oldv=torch.tensor([r['value'] for r in records],device='cuda');adv=ret-oldv;adv=(adv-adv.mean())/adv.std(unbiased=False).clamp_min(1e-6);oldlp=torch.tensor([r['log_prob'] for r in records],device='cuda');actions=torch.tensor([r['action'] for r in records],device='cuda');logs=[];early=False
 for epoch in range(epochs):
  order=torch.randperm(len(records)).tolist()
  for offset in range(0,len(records),16):
   ix=order[offset:offset+16];logits,v=model(*collate([records[i]['obs'] for i in ix]));dist=torch.distributions.Categorical(logits=logits);lp=dist.log_prob(actions[ix]);ratio=(lp-oldlp[ix]).exp();pl=-torch.minimum(ratio*adv[ix],ratio.clamp(.8,1.2)*adv[ix]).mean();vl=(v-ret[ix]).square().mean();entropy=dist.entropy().mean();loss=pl+.5*vl-.01*entropy
   assert bool(torch.isfinite(loss));opt.zero_grad();loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),.5);assert bool(torch.isfinite(norm));opt.step()
   with torch.no_grad():kl=((ratio-1)-(lp-oldlp[ix])).mean().item()
   logs.append([loss.item(),pl.item(),vl.item(),entropy.item(),kl,float(norm),float(((ratio-1).abs()>.2).float().mean())])
   if kl>.03:early=True;break
  if early:break
 torch.cuda.synchronize();return dict(zip(['loss','policy_loss','value_mse','entropy','approx_kl','grad_norm','clip_fraction'],np.mean(logs,0).tolist()),updates=len(logs),records=len(records),early_kl=early,seconds=time.perf_counter()-start)

def save(path,model,past,opt,iteration,seed,elapsed,logs,games):
 ck=dict(format='terminal-ppo-v1',model=model.state_dict(),past=past.state_dict(),optimizer=opt.state_dict(),iteration=iteration,seed=seed,elapsed=elapsed,logs=logs,games_gzip=gzip.compress(json.dumps(games,separators=(',',':')).encode(),mtime=0),python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),source_hashes=source_hashes())
 torch.save(ck,str(path)+'.tmp');os.replace(str(path)+'.tmp',path)

def train(seed,iterations=64,output=None,resume=None,cap=600,batch=16,save_steps=(16,32,64)):
 out=Path(output) if output else ROOT/f'seed-{seed}';out.mkdir(parents=True,exist_ok=True);seed_all(1400);model=TerminalNet().cuda();past=copy.deepcopy(model);opt=torch.optim.Adam(model.parameters(),lr=.0003);seed_all(seed);step=0;elapsed=0;logs=[];games=[]
 if resume:
  c=torch.load(resume,map_location='cpu',weights_only=False);assert c['seed']==seed;model.load_state_dict(c['model']);past.load_state_dict(c['past']);opt.load_state_dict(c['optimizer']);step=c['iteration'];elapsed=c['elapsed'];logs=c['logs'];games=c.get('games') if 'games' in c else json.loads(gzip.decompress(c['games_gzip']));random.setstate(c['python_rng']);np.random.set_state(c['numpy_rng']);torch.set_rng_state(c['torch_rng']);torch.cuda.set_rng_state_all(c['cuda_rng'])
 else:save(out/'untrained.pt',model,past,opt,0,seed,0,[],[])
 write(out/'config.json',dict(seed=seed,initial_seed=1400,iterations=iterations,batch_games=batch,save_steps=list(save_steps),cap_seconds=cap,torch=torch.__version__,parameters=sum(p.numel() for p in model.parameters()),source_hashes=source_hashes(),plan_sha256=sha(ROOT/'PLAN.md'),reward='terminal win/draw/loss, no shaping',resume=str(resume)))
 begin=time.perf_counter()
 while step<iterations and elapsed+time.perf_counter()-begin<cap:
  rec,gs,gen=collect(model,past,step,seed,batch);up=update(model,opt,rec);step+=1;games.extend(gs);logs.append(dict(iteration=step,generation=gen,update=up,opponents={m:float(np.mean([np.sign(g['scores'][g['learner_seat']]-g['scores'][1-g['learner_seat']]) for g in gs if g['opponent']==m])) for m in ['current','past','heuristic']}))
  if step%4==0:past.load_state_dict(model.state_dict())
  sec=elapsed+time.perf_counter()-begin
  if step in save_steps:save(out/f'iteration-{step}.pt',model,past,opt,step,seed,sec,logs,games)
  write(out/'curve.json',logs);print(seed,step,round(sec,2),up,flush=True)
 elapsed+=time.perf_counter()-begin;save(out/'latest.pt',model,past,opt,step,seed,elapsed,logs,games);write(out/'games.json',games);write(out/'result.json',dict(seed=seed,iterations=step,games=len(games),seconds=elapsed,complete=step==iterations,logs=logs));return out

class TerminalAgent:
 def __init__(self,path,device='cuda'):
  self.device=device;self.model=TerminalNet().to(device);self.model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model']);self.model.eval();self.times=[]
  with torch.no_grad():self.model(*collate([observe(State())[0]],device))
  if device=='cuda':torch.cuda.synchronize()
 def act(self,s,rng):
  start=time.perf_counter();obs,qs=observe(s)
  with torch.no_grad():logits,_=self.model(*collate([obs],self.device));a=logits[0].cpu().numpy();ix=int(rng.choice(np.flatnonzero(a>=a.max()-1e-7)))
  self.times.append(time.perf_counter()-start);return qs[ix].move if qs else None

def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=1401);p.add_argument('--iterations',type=int,default=64);p.add_argument('--output');p.add_argument('--resume');p.add_argument('--cap',type=float,default=600);p.add_argument('--batch',type=int,default=16);p.add_argument('--save-steps',type=int,nargs='+',default=[16,32,64]);a=p.parse_args();train(**vars(a))
if __name__=='__main__':main()
