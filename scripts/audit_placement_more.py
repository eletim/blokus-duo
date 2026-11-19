"""Independent replay, split, data, export and resume checks after timing runs."""
import copy,gzip,json,random,tempfile,subprocess,sys,time,concurrent.futures
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import placement_more as n
from audit_placement_scale import check_game
from check_placement_resume import same
R=n.R;OLD=n.OLD;e=n.e

def main():
 start=time.perf_counter();e.setup();idx=np.load(e.INDEX,mmap_mode='r');ex=set(n.read(R/'exclusions.json')['states']);labels=n.read(R/'train-labels.json');lookup={(r['seed'],r['game'],r['ply']):r for r in labels['labels']};allseen=set();total=0;checked=0
 with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool:
  paths=sorted((OLD/'games').glob('train-*.gz'))+sorted((R/'games').glob('train-*.gz'))
  for count,seen,ls in pool.map(check_game,[(p,None) for p in paths],chunksize=8):
   allseen.update(seen);total+=count
   for r in ls:assert lookup[(r['seed'],r['game'],r['ply'])]==r;checked+=1
  assert total==6144 and checked==len(lookup) and allseen==set(labels['visited']) and not set(labels['new_visited'])&ex
  devseen=set(n.read(OLD/'dev-labels.json')['visited']);assert not devseen&allseen and not any(e.contains(idx,h) for h in allseen)
  evalsets={};counts={}
  for phase in ('validation','confirmation'):
   paths=sorted(R.glob(f'{phase}-??-*.json.gz'))
   if not paths:continue
   bank=n.read(R/f'{phase}-bank.json')['prefixes'];seen=set();count=0
   for pr in bank:assert pr['state_hash']==n.state_hash(e.state(pr['moves'])) and pr['state_hash'] not in ex|allseen|devseen
   for c,vs,_ in pool.map(check_game,[(p,bank) for p in paths]):count+=c;seen.update(vs)
   assert count==576 and not seen&(ex|allseen|devseen) and not any(e.contains(idx,h) for h in seen);evalsets[phase]=seen;counts[phase]=count;total+=count
   out={p:n.aggregate(phase,p) for p in ('PF','QF','AF','AQ','AC','CF') if list(R.glob(f'{phase}-{p}-*.gz'))};assert out==n.read(R/f'{phase}-summary.json')
  if len(evalsets)==2:assert not evalsets['validation']&evalsets['confirmation']
 assert ('confirmation' in counts)==n.read(R/'selection.json')['passed']
 for path,h in n.read(R/'evaluation-manifest.json').items():assert e.sha(path)==h,path
 arr=np.load(R/'train-features.npz');count=Counter((r['seed'],r['game']) for r in labels['labels']);assert len(count)==6144
 assert np.array_equal(arr['x3'],np.array([r['target'] for r in labels['labels']],np.float32)) and np.array_equal(arr['x4'],np.array([1/count[r['seed'],r['game']] for r in labels['labels']],np.float32))
 # Re-encode a fixed distributed sample, including both old/new halves.
 sample=set(np.linspace(0,len(labels['labels'])-1,64,dtype=int).tolist());bygame={}
 for i in sample:
  r=labels['labels'][i];bygame.setdefault((r['seed'],r['game']),{})[r['ply']]=i
 for (seed,gid),plies in bygame.items():
  path=(OLD if gid<1024 else R)/'games'/f'train-{seed}-{gid:04d}.json.gz';g=json.load(gzip.open(path,'rt'));z=n.State()
  for ply,m in enumerate(g['moves']):
   if ply in plies:
    i=plies[ply];b,st=n.encode(z);assert np.array_equal(b[:4],arr['x0'][i]) and np.array_equal(st,arr['x1'][i]) and np.array_equal(n.mobility(z),arr['x2'][i])
   n.play(z,n.Move(*m) if m else None,validate=False)
 e.write(R/'audit.json',dict(pass_=True,training_games=6144,new_training_games=3072,evaluation_counts=counts,replayed_records=total,train_dev_overlap=0,train_evaluation_overlap=0,original_index_overlap=0,seconds=time.perf_counter()-start));print('AUDIT PASS',total,flush=True)

def checks():
 e.setup();exports=[];resumes=[];data=np.load(R/'train-features.npz');rows=len(data['x0']);bank=n.read(R/'validation-bank.json')['prefixes']
 for seed in e.SEEDS:
  p=n.PPOAgent(seed,'cpu',backend='native');before=copy.deepcopy(p.model.state_dict())
  for v in 'AC':
   ck=torch.load(R/f'{v}-{seed}.pt',map_location='cpu',weights_only=True);assert ck['data_sha']==e.sha(R/'train-features.npz') and ck['dev_sha']==e.sha(OLD/'dev-features.npz') and ck['updates']==((rows+63)//64)*16
   z=e.state(bank[0]['moves']);a=n.agent(v,p,500,2);n.play(z,a.act(z,random.Random(seed)),validate=True);assert all(torch.equal(t,before[k]) for k,t in p.model.state_dict().items());exports.append(dict(seed=seed,variant=v,pass_=True))
 p=n.PPOAgent(e.SEEDS[0],'cpu',backend='native');base=n.PlacementValue(p.model,'C');arrays={f'x{i}':data[f'x{i}'][-137:] for i in range(5)};arrays['x4']/=float(data['x4'].mean())
 with torch.no_grad():arrays['old']=base(*(torch.from_numpy(arrays[f'x{i}']) for i in range(3))).numpy()
 with tempfile.TemporaryDirectory() as td:
  td=Path(td);fixture=td/'fixture.npz';np.savez(fixture,**arrays)
  for device in ('cpu','cuda'):
   if device=='cuda' and not torch.cuda.is_available():continue
   for v in 'AC':
    src=R/f'resume-{v}-{e.SEEDS[0]}.pt';full=td/'full.pt';part=td/'part.pt';end=td/'end.pt'
    for epochs,a,b in [(6,src,full),(2,src,part),(4,part,end)]:subprocess.run([sys.executable,'scripts/check_placement_scale.py','worker',v,device,str(epochs),str(a),str(b),str(fixture)],check=True)
    assert same(torch.load(full,map_location='cpu',weights_only=False),torch.load(end,map_location='cpu',weights_only=False));resumes.append(dict(variant=v,device=device,pass_=True));print('RESUME',v,device,flush=True)
 e.write(R/'checks.json',dict(pass_=True,exports=exports,resumes=resumes));print('CHECKS PASS',flush=True)
if __name__=='__main__':checks() if len(sys.argv)>1 and sys.argv[1]=='checks' else main()
