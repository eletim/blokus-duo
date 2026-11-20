"""Real, resumable CUDA value learning on matched 24,576-game A/C data."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,copy,json,time,random
from pathlib import Path
import numpy as np
import torch
import placement_experiment as e
import placement_scale as scale
from blokus.placement_value import PlacementValue
from blokus.ppo_policy import PPOAgent
R=Path('artifacts/value-placement-24576')
def atomic_save(obj,path):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');torch.save(obj,tmp);tmp.replace(path)
def arrays(files):
 parts=[np.load(p) for p in files]
 try:return [np.concatenate([p[f'x{i}'] for p in parts]) for i in range(5)]
 finally:
  for p in parts:p.close()
def step(m,opt,x,old,ix):
 pred=m(*(a[ix] for a in x[:3]));loss=(x[4][ix]*((pred-x[3][ix]).square()+.1*(pred-old[ix]).square())).mean()
 assert torch.isfinite(loss),'nonfinite loss';opt.zero_grad();loss.backward()
 assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in m.parameters()),'nonfinite/missing gradient'
 opt.step();return float(loss.detach())
def rng_state(rng):return dict(numpy=rng.bit_generator.state,torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],python=random.getstate())
def restore_rng(state,rng):
 rng.bit_generator.state=state['numpy'];torch.set_rng_state(state['torch'].cpu());random.setstate(state['python'])
 if state['cuda']:torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])
def train():
 e.setup();assert torch.cuda.is_available();start=time.perf_counter();manifest=json.loads((R/'data-manifest.json').read_text())
 for p,h in manifest['file_shas'].items():assert e.sha(p)==h,p
 x=[torch.from_numpy(a).to('cuda') for a in arrays(manifest['train_files'])];v=[torch.from_numpy(a).to('cuda') for a in arrays(manifest['dev_files'])];x[4]/=x[4].mean()
 assert len(x[0])==manifest['old_labels']+manifest['new_labels'];runs=[];torch.cuda.reset_peak_memory_stats()
 cfg=dict(format='placement-large-resume-v1',plan_sha=e.sha(R/'PLAN.md'),data_manifest_sha=e.sha(R/'data-manifest.json'),model_code_sha=e.sha('blokus/placement_value.py'),lr=1e-4,batch=64,epochs=16,penalty=.1,samples=len(x[0]),train_games=24576)
 for seed in e.SEEDS:
  p=PPOAgent(seed,'cuda',backend='native');frozen=copy.deepcopy(p.model.state_dict());base=PlacementValue(p.model,'C').to('cuda')
  with torch.no_grad():old=torch.cat([base(*(a[k:k+2048] for a in x[:3])) for k in range(0,len(x[0]),2048)]).detach()
  del base
  for variant in 'AC':
   torch.manual_seed(seed);random.seed(seed);m=PlacementValue(p.model,variant).to('cuda');opt=torch.optim.Adam(m.parameters(),lr=1e-4);rng=np.random.default_rng(seed+7821);updates=0;epoch=0;wall=0.;checkpoint=R/f'resume-{variant}-{seed}.pt';run_cfg=dict(cfg,variant=variant,seed=seed,source_sha=e.sha(f'artifacts/ppo-gen3/seed-{seed}.pt'));curve=[]
   if checkpoint.exists():
    ck=torch.load(checkpoint,map_location='cuda',weights_only=False);assert ck['config']==run_cfg
    m.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);restore_rng(ck['rng'],rng);updates=ck['updates'];epoch=ck['epoch'];curve=ck['curve'];wall=ck['wall_seconds']
    assert all(g['lr']==1e-4 for g in opt.param_groups)
   else:
    curve=[dict(epoch=0,train_mse=scale.mse(m,x),dev_mse=scale.mse(m,v),updates=0,wall_seconds=0.)]
   # A completed checkpoint is the recovery source for its small inference exports.
   def export(ep):
    dest=R/f'epoch-{ep:02d}-{variant}-{seed}.pt'
    obj=dict(format='placement-value-v1',variant=variant,seed=seed,model={k:t.detach().cpu() for k,t in m.state_dict().items()},source_sha=run_cfg['source_sha'],train_games=24576,epochs=ep,updates=updates,config=run_cfg)
    if dest.exists():
     previous=torch.load(dest,map_location='cpu',weights_only=True);assert previous['updates']==updates and all(torch.equal(previous['model'][k],v) for k,v in obj['model'].items())
    else:atomic_save(obj,dest)
   if epoch:export(epoch)
   for ep in range(epoch+1,17):
    t=time.perf_counter();order=rng.permutation(len(x[0]));loss_sum=0.
    for k in range(0,len(order),64):loss_sum+=step(m,opt,x,old,order[k:k+64]);updates+=1
    assert all(torch.isfinite(t).all() for t in m.parameters());torch.cuda.synchronize();train_mse=scale.mse(m,x);dev_mse=scale.mse(m,v);wall+=time.perf_counter()-t
    curve.append(dict(epoch=ep,train_mse=train_mse,dev_mse=dev_mse,updates=updates,wall_seconds=wall,mean_minibatch_objective=loss_sum/((len(x[0])+63)//64)))
    ck=dict(config=run_cfg,model=m.state_dict(),optimizer=opt.state_dict(),rng=rng_state(rng),updates=updates,epoch=ep,curve=curve,wall_seconds=wall)
    atomic_save(ck,checkpoint);export(ep);e.write(R/f'learning-{variant}-{seed}.json',dict(config=run_cfg,curve=curve,updates=updates,train_labels=len(x[0]),dev_labels=len(v[0]),parameters=sum(t.numel() for t in m.parameters())))
    print('TRAIN',variant,seed,ep,updates,'dev',dev_mse,'wall',round(wall),flush=True)
   assert all(torch.equal(t,frozen[k]) for k,t in p.model.state_dict().items());runs.append(dict(variant=variant,seed=seed,updates=updates,epochs=16,wall_seconds=wall))
   e.write(R/f'learning-{variant}-{seed}.json',dict(config=run_cfg,curve=curve,updates=updates,train_labels=len(x[0]),dev_labels=len(v[0]),parameters=sum(t.numel() for t in m.parameters())))
 curves=[json.loads((R/f'learning-{v}-{s}.json').read_text())['curve'] for s in e.SEEDS for v in 'AC'];means=[float(np.mean([c[ep]['dev_mse'] for c in curves])) for ep in range(17)];chosen=min(range(1,17),key=lambda ep:(means[ep],ep));hashes={}
 for seed in e.SEEDS:
  for variant in 'AC':
   src=R/f'epoch-{chosen:02d}-{variant}-{seed}.pt';dest=R/f'{variant}-{seed}.pt'
   if dest.exists():assert dest.read_bytes()==src.read_bytes()
   else:dest.write_bytes(src.read_bytes())
   hashes[str(dest)]=e.sha(dest)
 e.write(R/'checkpoint-selection.json',dict(rule='one common epoch minimizing mean new-dev MSE across A/C and all three seeds; earliest exact tie',epoch=chosen,dev_mean_by_epoch=means,exports=hashes))
 e.write(R/'training-time.json',dict(seconds=time.perf_counter()-start,runs=runs,device='cuda',cuda_peak_bytes=torch.cuda.max_memory_allocated(),train_labels=len(x[0]),dev_labels=len(v[0]),resumed_wall_note='run wall sums include prior completed epochs; outer seconds cover this invocation'))
 print('TRAINED selected common epoch',chosen,flush=True)
if __name__=='__main__':train()
