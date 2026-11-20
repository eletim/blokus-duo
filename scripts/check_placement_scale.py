"""Post-evaluation fresh-process resume and all-export inference checks."""
import os,sys,tempfile,subprocess,copy,json,random
import numpy as np
import torch
import placement_scale as s
import placement_experiment as e
from check_placement_resume import same
from blokus.placement_value import PlacementValue,PlacementEvaluator
from blokus.ppo_policy import PPOAgent
from blokus.ppo_search_deeper import DeepSearch
from blokus.engine import play
R=s.R

def worker(variant,device,epochs,src,dst,fixture):
 e.setup();seed=e.SEEDS[0];p=PPOAgent(seed,device,backend='native');m=PlacementValue(p.model,variant).to(device);ck=torch.load(src,map_location=device,weights_only=False);m.load_state_dict(ck['model']);opt=torch.optim.Adam(m.parameters(),lr=1e-4);opt.load_state_dict(ck['optimizer']);rng=np.random.default_rng();rng.bit_generator.state=ck['rng'];data=np.load(fixture);x=[torch.tensor(data[f'x{i}'],device=device) for i in range(5)];old=torch.tensor(data['old'],device=device);updates=ck['updates']
 for epoch in range(int(epochs)):
  order=rng.permutation(len(x[0]))
  for k in range(0,len(order),64):
   ix=order[k:k+64];pred=m(*(a[ix] for a in x[:3]));loss=(x[4][ix]*((pred-x[3][ix]).square()+.1*(pred-old[ix]).square())).mean();opt.zero_grad();loss.backward();opt.step();updates+=1
 torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),rng=rng.bit_generator.state,updates=updates),dst)

def main():
 e.setup();result=[];exports=[]
 feature_sha={p.name:e.sha(p) for p in R.glob('*-features.npz')}
 from collections import Counter
 for phase in ('train','dev'):
  ar=np.load(R/f'{phase}-features.npz');labels=json.loads((R/f'{phase}-labels.json').read_text())['labels'];counts=Counter((r['seed'],r['game']) for r in labels)
  assert all(len(ar[f'x{i}'])==len(labels) for i in range(6))
  assert np.array_equal(ar['x3'],np.array([r['target'] for r in labels],np.float32))
  assert np.array_equal(ar['x5'],np.array([r['game'] for r in labels],np.float32))
  assert np.array_equal(ar['x4'],np.array([1/counts[(r['seed'],r['game'])] for r in labels],np.float32))
 for size in s.STAGES:
  bank=json.loads((R/f'n{size}'/'validation-bank.json').read_text())['prefixes']
  for seed in e.SEEDS:
   p=PPOAgent(seed,'cpu',backend='native');before=copy.deepcopy(p.model.state_dict())
   for variant in 'ABC':
    ck=torch.load(R/f'n{size}'/f'{variant}-{seed}.pt',map_location='cpu',weights_only=True);assert ck['source_sha']==e.sha(f'artifacts/ppo-gen3/seed-{seed}.pt') and ck['train_games']==size and ck['epochs']==16 and ck['data_shas']==feature_sha
    rows=sum(1 for r in json.loads((R/'train-labels.json').read_text())['labels'] if r['game']<size//3);assert ck['updates']==((rows+63)//64)*16
    model=PlacementValue(p.model,variant);model.load_state_dict(ck['model']);a=DeepSearch(p,500,max_depth=2,value_evaluator=PlacementEvaluator(model));z=e.state(bank[0]['moves']);m=a.act(z,random.Random(seed));play(z,m,validate=True);assert all(torch.equal(t,before[k]) for k,t in p.model.state_dict().items());exports.append(dict(size=size,variant=variant,seed=seed,pass_=True))
 p=PPOAgent(e.SEEDS[0],'cpu',backend='native');base=PlacementValue(p.model,'C');data=np.load(R/'train-features.npz');arrays={f'x{i}':data[f'x{i}'][:137] for i in range(5)}
 arrays['x4']=arrays['x4']/float(data['x4'].mean())
 with torch.no_grad():arrays['old']=base(*(torch.from_numpy(arrays[f'x{i}']) for i in range(3))).numpy()
 with tempfile.TemporaryDirectory() as td:
  from pathlib import Path
  td=Path(td);fixture=td/'fixture.npz';np.savez(fixture,**arrays)
  for device in ('cpu','cuda'):
   if device=='cuda' and not torch.cuda.is_available():continue
   for variant in 'ABC':
    src=R/'n3072'/f'resume-{variant}-{e.SEEDS[0]}.pt';full=td/'full.pt';part=td/'part.pt';end=td/'end.pt'
    for epochs,a,b in [(6,src,full),(2,src,part),(4,part,end)]:subprocess.run([sys.executable,__file__,'worker',variant,device,str(epochs),str(a),str(b),str(fixture)],check=True)
    assert same(torch.load(full,map_location='cpu',weights_only=False),torch.load(end,map_location='cpu',weights_only=False));result.append(dict(variant=variant,device=device,pass_=True));print('RESUME PASS',variant,device,flush=True)
 e.write(R/'export-resume-check.json',dict(exports=exports,resume=result,fixture_states=137,comparison='actual final Adam/RNG checkpoint; 6 vs2+freshprocessrestore+4 fixture epochs; equality includes weights optimizer RNG',pass_=True))
if __name__=='__main__':
 if len(sys.argv)>1:worker(*sys.argv[2:])
 else:main()
