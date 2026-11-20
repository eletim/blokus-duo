"""Actual new training step: fresh-process Adam/RNG continuation and CPU exports."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys,subprocess,tempfile,copy,random,json
from pathlib import Path
import numpy as np
import torch
import placement_large_train as t
from check_placement_resume import same
from blokus.placement_value import PlacementValue
from blokus.ppo_policy import PPOAgent
R=t.R

def worker(device,variant,src,dst,fixture,epochs):
 t.e.setup();p=PPOAgent(t.e.SEEDS[0],device,backend='native');m=PlacementValue(p.model,variant).to(device);ck=torch.load(src,map_location=device,weights_only=False);m.load_state_dict(ck['model']);opt=torch.optim.Adam(m.parameters(),lr=1e-4);opt.load_state_dict(ck['optimizer']);rng=np.random.default_rng();t.restore_rng(ck['rng'],rng);data=np.load(fixture);x=[torch.from_numpy(data[f'x{i}']).to(device) for i in range(5)];old=torch.from_numpy(data['old']).to(device);updates=ck['updates']
 for _ in range(int(epochs)):
  order=rng.permutation(len(x[0]))
  for k in range(0,len(order),64):t.step(m,opt,x,old,order[k:k+64]);updates+=1
 t.atomic_save(dict(model=m.state_dict(),optimizer=opt.state_dict(),rng=t.rng_state(rng),updates=updates),dst)

def check():
 t.e.setup();p=PPOAgent(t.e.SEEDS[0],'cpu',backend='native');data=np.load('artifacts/value-placement-6144/train-features.npz');x={f'x{i}':data[f'x{i}'][-137:].copy() for i in range(5)};x['x4']/=float(data['x4'].mean())
 with torch.no_grad():
  a=PlacementValue(p.model,'A');c=PlacementValue(p.model,'C');args=[torch.from_numpy(x[f'x{i}']) for i in range(3)];assert torch.equal(a(*args),c(*args));x['old']=c(*args).numpy()
 checks=[]
 with tempfile.TemporaryDirectory() as tmp:
  tmp=Path(tmp);fixture=tmp/'fixture.npz';np.savez(fixture,**x)
  for device in ('cpu','cuda'):
   assert device!='cuda' or torch.cuda.is_available()
   for variant in 'AC':
    torch.manual_seed(987);random.seed(987);m=PlacementValue(p.model,variant).to(device);opt=torch.optim.Adam(m.parameters(),lr=1e-4);rng=np.random.default_rng(987);src=tmp/'start.pt';actual=R/f'resume-{variant}-{t.e.SEEDS[0]}.pt'
    if actual.exists():ck=torch.load(actual,map_location=device,weights_only=False);m.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);t.restore_rng(ck['rng'],rng);updates=ck['updates']
    else:updates=0
    t.atomic_save(dict(model=m.state_dict(),optimizer=opt.state_dict(),rng=t.rng_state(rng),updates=updates),src)
    full=tmp/'full.pt';part=tmp/'part.pt';end=tmp/'end.pt'
    for ep,left,right in [(3,src,full),(1,src,part),(2,part,end)]:subprocess.run([sys.executable,__file__,'worker',device,variant,str(left),str(right),str(fixture),str(ep)],check=True)
    f=torch.load(full,map_location='cpu',weights_only=False);g=torch.load(end,map_location='cpu',weights_only=False);assert same(f,g);initial=torch.load(src,map_location='cpu',weights_only=False);assert any(not torch.equal(initial['model'][k],v) for k,v in f['model'].items());assert f['updates']==updates+9
    checks.append(dict(device=device,variant=variant,seed=t.e.SEEDS[0],bit_exact_resume=True,real_weight_change=True,actual_training_checkpoint=actual.exists(),optimizer_steps=9));print('RESUME PASS',device,variant,flush=True)
 t.e.write(R/('checks-post-training.json' if (R/'checkpoint-selection.json').exists() else 'checks-pre-training.json'),dict(pass_=True,initial_A_C_bit_equal=True,fixture_states=137,checks=checks))
if __name__=='__main__':worker(*sys.argv[2:]) if len(sys.argv)>1 else check()
