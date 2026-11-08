"""Independent-process Adam/RNG continuation and full production endpoint replay."""
from h_terminal_value import *

def check(mode):
 setup();p=PPOAgent(SEEDS[0],'cpu');head=copy.deepcopy(p.model.value);opt=torch.optim.Adam(head.parameters(),lr=1e-4);rng=np.random.default_rng(991)
 x=torch.from_numpy(np.random.default_rng(50).normal(size=(80,81)).astype('float32'));y=torch.from_numpy(np.random.default_rng(51).choice([-1.,0.,1.],size=80).astype('float32'))
 if mode=='resume':
  c=torch.load(R/'check-first.pt',weights_only=False);head.load_state_dict(c['head']);opt.load_state_dict(c['opt']);rng.bit_generator.state=c['rng']
 for _ in range({'full':10,'first':4,'resume':6}[mode]):
  ix=rng.permutation(80)[:32];loss=((head(x[ix]).squeeze(-1)-y[ix])**2).mean();opt.zero_grad();loss.backward();opt.step()
 torch.save(dict(head=head.state_dict(),opt=opt.state_dict(),rng=rng.bit_generator.state),R/f'check-{mode}.pt')

def exact(a,b):
 if isinstance(a,torch.Tensor):return torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
 return a==b

def saved_dataset(p):
 # Reproduce from saved selected labels, without the external historical index.
 rows=json.loads((R/f'learning-{p.model_seed}.json').read_text())['train_labels']
 selected={(r['game'],r['ply']):r for r in rows};data=[]
 for g in json.loads((R/f'train-{p.model_seed}.json').read_text())['games']:
  s=State();local=[]
  for ply,m in enumerate(g['moves']):
   if (g['id'],ply) in selected:
    label=selected[(g['id'],ply)];assert label['state_hash']==state_hash(s) and label['turn']==s.turn and label['target']==outcome(g['scores'],s.turn)
    local.append((features(p,s),label['target']))
   play(s,Move(*m) if m else None,validate=True)
  assert local
  data.extend((x,y,1/len(local)) for x,y in local)
 assert len(data)==len(rows)
 return data

def endpoints():
 setup();checks=[]
 for seed in SEEDS:
  p=PPOAgent(seed,'cpu');data=saved_dataset(p);x,y,w=tensor_data(data);w/=w.mean();head=copy.deepcopy(p.model.value);opt=torch.optim.Adam(head.parameters(),lr=1e-4);rng=np.random.default_rng(seed+5100)
  with torch.no_grad():old=head(x).squeeze(-1).clone()
  for ep in range(8):
   ix=rng.permutation(len(x))
   for start in range(0,len(ix),64):
    z=ix[start:start+64];v=head(x[z]).squeeze(-1);loss=(w[z]*((v-y[z])**2+.1*(v-old[z])**2)).mean();opt.zero_grad();loss.backward();opt.step()
  ck=torch.load(R/f'training-state-{seed}.pt',weights_only=False)
  assert exact(ck['head'],head.state_dict()) and exact(ck['optimizer'],opt.state_dict()) and exact(ck['rng'],rng.bit_generator.state)
  q=PPOAgent(seed,'cpu',checkpoint=R/f'student-{seed}.pt');assert exact(head.state_dict(),q.model.value.state_dict());checks.append(seed)
 write(R/'update-replay.json',dict(passed=True,seeds=checks,full_updates_and_adam_reproduced=True))
 print('All training updates, Adam and export replay PASS',flush=True)

if __name__=='__main__':
 mode=sys.argv[1]
 if mode=='endpoints':endpoints()
 elif mode=='compare':
  assert exact(torch.load(R/'check-full.pt',weights_only=False),torch.load(R/'check-resume.pt',weights_only=False));write(R/'resume-check.json',dict(passed=True,updates=10,split=[4,6],separate_processes=True));print('Resume exact PASS')
 else:check(mode)
