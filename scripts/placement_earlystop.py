"""Separate independent test of development-selected stopping; no production changes."""
import argparse,copy,gzip,json,time,random,concurrent.futures,tempfile,subprocess,sys
from pathlib import Path
import placement_scale as s
import placement_experiment as e
import numpy as np
import torch
from blokus.placement_value import PlacementValue,PlacementEvaluator
from blokus.ppo_policy import PPOAgent
from blokus.ppo_search_deeper import DeepSearch
from blokus.engine import Move,play
from blokus.ppo_runtime import state_hash
R=s.R/'earlystop'
PAIRS=[('A','F'),('B','F'),('C','F'),('A','C'),('B','C')]

def train():
 e.setup();R.mkdir(exist_ok=True);start=time.perf_counter();device='cuda';assert torch.cuda.is_available();tr=np.load(s.R/'train-features.npz');dv=np.load(s.R/'dev-features.npz');x=[torch.tensor(tr[f'x{i}'],device=device) for i in range(5)];v=[torch.tensor(dv[f'x{i}'],device=device) for i in range(5)];x[4]/=x[4].mean();choices=json.loads((s.R/'earlystop-epochs.json').read_text());checks=[]
 for variant in 'ABC':
  curves=[json.loads((s.R/'n3072'/f'learning-{variant}-{seed}.json').read_text())['curve'] for seed in e.SEEDS];means=np.mean([[row['dev_mse'] for row in curve] for curve in curves],axis=0);assert int(np.argmin(means))==choices[variant]['epoch']
 for seed in e.SEEDS:
  p=PPOAgent(seed,device,backend='native');frozen=copy.deepcopy(p.model.state_dict());base=PlacementValue(p.model,'C').to(device)
  with torch.no_grad():old=torch.cat([base(*(a[k:k+2048] for a in x[:3])) for k in range(0,len(x[0]),2048)]).detach()
  for variant in 'ABC':
   assert not (R/f'{variant}-{seed}.pt').exists();epoch_count=choices[variant]['epoch'];torch.manual_seed(seed);m=PlacementValue(p.model,variant).to(device);opt=torch.optim.Adam(m.parameters(),lr=1e-4);rng=np.random.default_rng(seed+7821);updates=0
   s.mse(m,x);s.mse(m,v)
   for epoch in range(epoch_count):
    order=rng.permutation(len(x[0]))
    for k in range(0,len(order),64):
     ix=order[k:k+64];pred=m(*(a[ix] for a in x[:3]));loss=(x[4][ix]*((pred-x[3][ix]).square()+.1*(pred-old[ix]).square())).mean();assert torch.isfinite(loss)
     opt.zero_grad();loss.backward();assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in m.parameters());opt.step();updates+=1
    s.mse(m,x);s.mse(m,v)
   train_mse=s.mse(m,x);dev_mse=s.mse(m,v);expected=json.loads((s.R/'n3072'/f'learning-{variant}-{seed}.json').read_text())['curve'][epoch_count];assert abs(train_mse-expected['train_mse'])<1e-7 and abs(dev_mse-expected['dev_mse'])<1e-7
   assert all(torch.equal(t,frozen[k]) for k,t in p.model.state_dict().items())
   torch.save(dict(format='placement-value-v1',variant=variant,seed=seed,model={k:t.detach().cpu() for k,t in m.state_dict().items()},source_sha=e.sha(f'artifacts/ppo-gen3/seed-{seed}.pt'),data_shas={p.name:e.sha(p) for p in s.R.glob('*-features.npz')},train_games=3072,epochs=epoch_count,updates=updates),R/f'{variant}-{seed}.pt')
   torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),rng=rng.bit_generator.state,updates=updates),R/f'resume-{variant}-{seed}.pt')
   checks.append(dict(variant=variant,seed=seed,epochs=epoch_count,updates=updates,train_mse=train_mse,dev_mse=dev_mse,train_difference=train_mse-expected['train_mse'],dev_difference=dev_mse-expected['dev_mse']));print('EARLY TRAIN',checks[-1],flush=True)
 e.write(R/'training-check.json',dict(seconds=time.perf_counter()-start,checks=checks,policy_unchanged=True));print('EARLY TRAINING COMPLETE',flush=True)

def banks():
 ex=s.exclusions();idx=np.load(e.INDEX,mmap_mode='r')
 for phase in ('train','dev'):ex.update(json.loads((s.R/f'{phase}-labels.json').read_text())['visited'])
 for size,phase in [(384,'curve'),(1536,'curve'),(3072,'validation'),(3072,'confirmation')]:
  d=s.R/f'n{size}';bank=json.loads((d/f'{phase}-bank.json').read_text())['prefixes'];ex.update(pr['state_hash'] for pr in bank)
  for path in d.glob(f'{phase}-??-*.json.gz'):
   for g in json.load(gzip.open(path,'rt'))['games']:
    z=e.state(bank[g['prefix']]['moves']);ex.add(state_hash(z))
    for m in g['moves']:play(z,Move(*m) if m else None,validate=False);ex.add(state_hash(z))
 e.write(R/'excluded-states.json',dict(states=sorted(ex)))
 rng=random.Random(696000000);bank=[]
 for i in range(32):
  pr=e.prefix(rng,idx,ex,n=8+(i%3)*2);ex.add(pr['state_hash']);bank.append(pr)
 e.write(R/'confirmation-bank.json',dict(prefixes=bank))
 files=['scripts/placement_earlystop.py','scripts/placement_experiment.py','scripts/placement_scale.py','blokus/placement_value.py','blokus/ppo_search_deeper.py']+[str(p) for p in R.glob('[ABC]-*.pt')]+[str(R/'confirmation-bank.json')]
 e.write(R/'evaluation-manifest.json',{p:e.sha(p) for p in files});print('EARLY MODELS/BANK FROZEN',flush=True)

def aggregate(pair):
 e.R=R;out=e.aggregate('confirmation',pair);rows=[]
 for seed in e.SEEDS:rows+=json.load(gzip.open(R/f'confirmation-{pair}-{seed}.json.gz','rt'))['games']
 groups=sorted({g['prefix'] for g in rows});pts=np.array([np.mean([g['point'] for g in rows if g['prefix']==i]) for i in groups]);diff=np.array([np.mean([g['difference'] for g in rows if g['prefix']==i]) for i in groups]);rng=np.random.default_rng(771);ix=rng.integers(len(groups),size=(20000,len(groups)))
 out['point_adjusted_ci']=np.quantile(pts[ix].mean(1),[.00625,.99375]).tolist();out['difference_adjusted_ci']=np.quantile(diff[ix].mean(1),[.00625,.99375]).tolist();out['ci_level']=.9875;out['ci_note']='four primary comparisons Bonferroni; CF descriptive';return out

def summarize():
 out={a+b:aggregate(a+b) for a,b in PAIRS};passed=[]
 for variant in 'AB':
  if all(out[variant+b]['point_adjusted_ci'][0]>.5+1e-9 and out[variant+b]['difference']>0 and out[variant+b]['stats'][variant]['over_550']<=.01 for b in 'FC'):passed.append(variant)
 choice=max(passed,key=lambda a:(2*out[a+'F']['wins']+out[a+'F']['draws'],a=='A')) if passed else None
 e.write(R/'decision.json',dict(results=out,passed=passed,selected=choice));print('EARLY DECISION',choice,{k:round(v['point'],4) for k,v in out.items()},flush=True)

def evaluate():
 e.R=R;e.evaluate('confirmation',PAIRS);summarize()

def audit():
 from audit_placement_scale import check_game
 start=time.perf_counter();idx=np.load(e.INDEX,mmap_mode='r');ex=set(json.loads((R/'excluded-states.json').read_text())['states']);bank=json.loads((R/'confirmation-bank.json').read_text())['prefixes'];starts=set()
 for pr in bank:
  h=state_hash(e.state(pr['moves']));assert h==pr['state_hash'] and h not in ex|starts and not e.contains(idx,h);starts.add(h)
 seen=set();count=0
 with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool:
  for n,visited,labels in pool.map(check_game,[(p,bank) for p in sorted(R.glob('confirmation-??-*.json.gz'))]):count+=n;seen.update(visited)
 assert count==960 and not seen&ex and not any(e.contains(idx,h) for h in seen)
 for path,h in json.loads((R/'evaluation-manifest.json').read_text()).items():assert e.sha(path)==h,path
 decision=json.loads((R/'decision.json').read_text());assert decision['results']=={a+b:aggregate(a+b) for a,b in PAIRS}
 e.write(R/'audit.json',dict(pass_=True,games=count,historical_and_current_training_evaluation_overlap=0,original_training_overlap=0,seconds=time.perf_counter()-start));print('EARLY AUDIT PASS',count,flush=True)

def check():
 from check_placement_resume import same
 e.setup();rows=[];resumes=[];data=np.load(s.R/'train-features.npz');p=PPOAgent(e.SEEDS[0],'cpu',backend='native');base=PlacementValue(p.model,'C');arrays={f'x{i}':data[f'x{i}'][:137] for i in range(5)};arrays['x4']/=float(data['x4'].mean())
 with torch.no_grad():arrays['old']=base(*(torch.from_numpy(arrays[f'x{i}']) for i in range(3))).numpy()
 for seed in e.SEEDS:
  p=PPOAgent(seed,'cpu',backend='native');before=copy.deepcopy(p.model.state_dict())
  for variant in 'ABC':
   ck=torch.load(R/f'{variant}-{seed}.pt',map_location='cpu',weights_only=True);assert ck['source_sha']==e.sha(f'artifacts/ppo-gen3/seed-{seed}.pt');m=PlacementValue(p.model,variant);m.load_state_dict(ck['model']);z=e.state(json.loads((R/'confirmation-bank.json').read_text())['prefixes'][0]['moves']);a=DeepSearch(p,500,max_depth=2,value_evaluator=PlacementEvaluator(m));move=a.act(z,random.Random(seed));play(z,move,validate=True);assert all(torch.equal(t,before[k]) for k,t in p.model.state_dict().items());rows.append(dict(seed=seed,variant=variant,pass_=True))
 with tempfile.TemporaryDirectory() as td:
  td=Path(td);fixture=td/'fixture.npz';np.savez(fixture,**arrays)
  for device in ('cpu','cuda'):
   if device=='cuda' and not torch.cuda.is_available():continue
   for variant in 'ABC':
    src=R/f'resume-{variant}-{e.SEEDS[0]}.pt';full=td/'full.pt';part=td/'part.pt';end=td/'end.pt'
    for epochs,a,b in [(6,src,full),(2,src,part),(4,part,end)]:subprocess.run([sys.executable,'scripts/check_placement_scale.py','worker',variant,device,str(epochs),str(a),str(b),str(fixture)],check=True)
    assert same(torch.load(full,map_location='cpu',weights_only=False),torch.load(end,map_location='cpu',weights_only=False));resumes.append(dict(variant=variant,device=device,pass_=True));print('EARLY RESUME PASS',variant,device,flush=True)
 e.write(R/'export-resume-check.json',dict(exports=rows,resume=resumes,pass_=True))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('command',choices=['train','banks','evaluate','summarize','audit','check']);args=ap.parse_args();globals()[args.command]()
