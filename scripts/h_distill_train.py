"""Policy-head-only teacher imitation, cached frozen CPU features, CUDA Adam."""
import sys,copy,json,time,random,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from h_distill import *

def cache(agent,rows):
 result=[];capture=[]
 hook=agent.model.policy.register_forward_pre_hook(lambda mod,args:capture.append(args[0].detach().clone()))
 try:
  for row in rows:
   s=state(row['moves']);obs,qs=agent._observation(s);target=np.zeros(len(qs),np.float32)
   for m in row['targets']:
    target[[raw(q.move) for q in qs].index(m)]=1/len(row['targets'])
   with torch.no_grad():l,v=agent.model(*collate([obs],'cpu'))
   parts=capture.pop()[0];prior=torch.as_tensor(obs[2][:,:12])@agent.model.prior
   with torch.no_grad():other=agent.model.policy(parts).squeeze(-1)+prior
   torch.testing.assert_close(other,l[0],atol=2e-6,rtol=1e-6)
   result.append((parts,prior,l[0].detach(),torch.from_numpy(target),float(v)))
 finally:hook.remove()
 return result

def batch(data,indices,device):
 items=[data[int(i)] for i in indices];n=max(len(t[0]) for t in items);b=len(items)
 parts=torch.zeros(b,n,144);prior=torch.zeros(b,n);original=torch.zeros(b,n);target=torch.zeros(b,n);mask=torch.zeros(b,n,dtype=torch.bool)
 for i,(x,p,l,t,v) in enumerate(items):
  k=len(x);parts[i,:k]=x;prior[i,:k]=p;original[i,:k]=l;target[i,:k]=t;mask[i,:k]=True
 return tuple(t.to(device) for t in (parts,prior,original,target,mask))

def loss(head,items):
 x,prior,old,target,mask=items;logits=head(x).squeeze(-1)+prior
 lp=torch.log_softmax(logits.masked_fill(~mask,-1e9),-1);oldlp=torch.log_softmax(old.masked_fill(~mask,-1e9),-1)
 ce=-(target*lp).sum(-1).mean();kl=(oldlp.exp()*(oldlp-lp)*mask).sum(-1).mean()
 return ce+.05*kl,ce,kl,logits.masked_fill(~mask,-float('inf'))

def metrics(head,data,device):
 total=[];correct=[];oldcorrect=[]
 with torch.no_grad():
  for start in range(0,len(data),16):
   items=batch(data,range(start,min(start+16,len(data))),device);_,ce,kl,l=loss(head,items);target=items[3];old=items[2].masked_fill(~items[4],-float('inf'))
   correct.extend((target.gather(1,l.argmax(-1)[:,None])[:,0]>0).cpu().tolist());oldcorrect.extend((target.gather(1,old.argmax(-1)[:,None])[:,0]>0).cpu().tolist());total.append([float(ce),float(kl),len(l)])
 return dict(ce=sum(a*n for a,b,n in total)/len(data),kl=sum(b*n for a,b,n in total)/len(data),agreement=float(np.mean(correct)),original_agreement=float(np.mean(oldcorrect)),labels=len(data))

class Trainer:
 def __init__(self,seed,data,device='cuda'):
  self.agent=PPOAgent(seed,'cpu');self.device=device;self.data=data;self.rng=np.random.default_rng(seed-1751400+9300)
  self.head=copy.deepcopy(self.agent.model.policy).to(device);self.optimizer=torch.optim.Adam(self.head.parameters(),lr=1e-4);self.updates=0;self.epoch=0;self.position=0;self.order=[]
 def step(self):
  if self.position>=len(self.order):self.order=self.rng.permutation(len(self.data)).tolist();self.position=0;self.epoch+=1
  ix=self.order[self.position:self.position+16];self.position+=len(ix);self.optimizer.zero_grad(set_to_none=True)
  total,ce,kl,_=loss(self.head,batch(self.data,ix,self.device));assert torch.isfinite(total)
  total.backward();norm=sum(float(p.grad.square().sum()) for p in self.head.parameters());assert np.isfinite(norm)
  self.optimizer.step();assert all(torch.isfinite(p).all() for p in self.head.parameters());self.updates+=1;return dict(loss=float(total),ce=float(ce),kl=float(kl),gradient_norm=norm**.5,zero_gradient=norm==0,legal_counts=[len(self.data[int(i)][0]) for i in ix])
 def save(self,path):
  temporary=Path(str(path)+'.tmp')
  torch.save(dict(head=self.head.state_dict(),optimizer=self.optimizer.state_dict(),rng=self.rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],updates=self.updates,epoch=self.epoch,position=self.position,order=self.order),temporary);temporary.replace(path)
 def load(self,path):
  ck=torch.load(path,map_location=self.device,weights_only=False);self.head.load_state_dict(ck['head']);self.optimizer.load_state_dict(ck['optimizer']);self.rng.bit_generator.state=ck['rng'];torch.set_rng_state(ck['torch_rng'].cpu())
  if ck['cuda_rng']:torch.cuda.set_rng_state_all([x.cpu() for x in ck['cuda_rng']])
  for k in ('updates','epoch','position','order'):setattr(self,k,ck[k])

def train(seed,resume=False):
 started=time.perf_counter();p=PPOAgent(seed,'cpu');trainrows=json.loads((R/f'train-{seed}.json').read_text())['labels'];devrows=json.loads((R/f'dev-{seed}.json').read_text())['labels']
 data=cache(p,trainrows);dev=cache(p,devrows);t=Trainer(seed,data);before=metrics(t.head,dev,'cuda');logs=[]
 if resume:
  t.load(R/f'resume-{seed}.pt');assert t.position==len(t.order), 'main resume checkpoints are epoch boundaries'
  logs=json.loads((R/f'curve-{seed}.json').read_text());assert len(logs)==t.epoch
 for ep in range(t.epoch,8):
  updates=[t.step() for _ in range((len(data)+15)//16)];torch.cuda.synchronize()
  logs.append(dict(epoch=ep+1,updates=t.updates,train=metrics(t.head,data,'cuda'),dev=metrics(t.head,dev,'cuda'),last_update=updates[-1],zero_gradient_batches=sum(u['zero_gradient'] for u in updates),wall_seconds=time.perf_counter()-started));print(seed,logs[-1],flush=True)
  t.save(R/f'resume-{seed}.pt');write(R/f'curve-{seed}.json',logs)
 p.model.policy.load_state_dict({k:v.cpu() for k,v in t.head.state_dict().items()})
 src=torch.load(f'artifacts/ppo-gen3/seed-{seed}.pt',map_location='cpu',weights_only=True)
 changed=[k for k,v in p.model.state_dict().items() if not torch.equal(v,src['model'][k])];assert changed and all(k.startswith('policy.') for k in changed)
 ck=dict(format='terminal-ppo-inference-v2',channels=4,model=p.model.state_dict(),research_name='H-policy-distillation',teacher_sha256=sha(f'artifacts/ppo-gen3/seed-{seed}.pt'),seed=seed,epochs=8,updates=t.updates,training_data_sha256=sha(R/f'train-{seed}.json'))
 torch.save(ck,R/f'student-{seed}.pt');write(R/f'learning-{seed}.json',dict(before_dev=before,curve=logs,changed_parameters=changed,wall_seconds=time.perf_counter()-started,device=torch.cuda.get_device_name(),torch_version=torch.__version__))

if __name__=='__main__':
 setup();p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=0);p.add_argument('--resume',action='store_true');args=p.parse_args()
 for seed in ([args.seed] if args.seed else SEEDS):train(seed,args.resume)
