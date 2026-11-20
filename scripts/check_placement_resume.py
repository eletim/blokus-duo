"""Restore actual optimizer/RNG checkpoints in fresh CPU/CUDA processes."""
from placement_experiment import *
import subprocess,sys,tempfile

def worker(variant,device,epochs,src,dst,fixture):
 setup();seed=SEEDS[0];p=PPOAgent(seed,device,backend='native');m=PlacementValue(p.model,variant).to(device);ck=torch.load(src,map_location=device,weights_only=False);m.load_state_dict(ck['model']);opt=torch.optim.Adam(m.parameters(),lr=1e-4);opt.load_state_dict(ck['optimizer']);rng=np.random.default_rng();rng.bit_generator.state=ck['rng'];data=np.load(fixture);x=[torch.tensor(data[k],device=device) for k in ('b','s','e','y')];old=torch.tensor(data['old'],device=device);updates=ck['updates']
 for epoch in range(int(epochs)):
  order=rng.permutation(len(x[0]))
  for offset in range(0,len(order),64):
   ix=order[offset:offset+64];out=m(*(a[ix] for a in x[:3]));loss=((out-x[3][ix]).square()+.1*(out-old[ix]).square()).mean();opt.zero_grad();loss.backward();opt.step();updates+=1
 torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),rng=rng.bit_generator.state,updates=updates),dst)

def same(a,b):
 if isinstance(a,torch.Tensor):return torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
 return a==b

def check():
 setup();p=PPOAgent(SEEDS[0],'cpu',backend='native');rows=[];d=json.load(gzip.open(R/f'train-{SEEDS[0]}.json.gz','rt'));base=PlacementEvaluator(PlacementValue(p.model,'C'))
 for g in d['games'][:5]:
  s=State()
  for i,m in enumerate(g['moves']):
   if i>=g['prefix_length']:
    b,st=encode(s);rows.append((b[:4],st,mobility(s),float(np.sign(g['scores'][s.turn]-g['scores'][1-s.turn])),base(s)))
   play(s,Move(*m) if m else None,validate=False)
 result=[]
 with tempfile.TemporaryDirectory() as td:
  fixture=Path(td)/'fixture.npz';np.savez(fixture,**{k:np.stack([r[i] for r in rows]).astype(np.float32) for i,k in enumerate(('b','s','e','y','old'))})
  for device in ('cpu','cuda'):
   if device=='cuda' and not torch.cuda.is_available():continue
   for variant in 'ABC':
    src=R/f'resume-{variant}-{SEEDS[0]}.pt';full=Path(td)/'full.pt';part=Path(td)/'part.pt';end=Path(td)/'end.pt'
    for epochs,a,b in [(6,src,full),(2,src,part),(4,part,end)]:subprocess.run([sys.executable,__file__,'worker',variant,device,str(epochs),str(a),str(b),str(fixture)],check=True)
    assert same(torch.load(full,map_location='cpu',weights_only=False),torch.load(end,map_location='cpu',weights_only=False));result.append(dict(variant=variant,device=device,pass_=True));print('resume',variant,device,'PASS',flush=True)
 write(R/'resume-check.json',dict(results=result,fixture_states=len(rows),comparison='6 epochs versus 2 + fresh-process restore + 4; actual saved Adam and RNG',pass_=True))
if __name__=='__main__':
 if len(sys.argv)>1:worker(*sys.argv[2:])
 else:check()
