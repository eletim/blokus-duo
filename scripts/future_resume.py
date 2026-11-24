from future_common import *
from blokus.future_occupancy import FutureValue
from future_train import tensors,loss
import subprocess,sys,argparse,tempfile

def child(arm,seed,device,out):
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);_,v=load(seed,'native');m=FutureValue(v.model,arm).to(device);ck=torch.load(R/'models'/f'{arm}-{seed}'/'epoch-8.pt',map_location='cpu',weights_only=False);m.load_state_dict(ck['model']);opt=torch.optim.Adam(m.parameters(),lr=1e-4);opt.load_state_dict(ck['optimizer']);torch.set_rng_state(ck['torch_rng'])
 if device=='cuda':torch.cuda.set_rng_state(ck['cuda_rng'])
 d=tensors('dev',device);opt.zero_grad();l,_,_=loss(m,d,slice(0,128));l.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step();torch.save(dict(model={k:t.detach().cpu() for k,t in m.state_dict().items()},loss=float(l)),out)
def main():
 rows=[]
 with tempfile.TemporaryDirectory() as tmp:
  for seed in SEEDS:
   for arm in 'ABC':
    for device in ['cpu','cuda']:
     outputs=[]
     for i in [0,1]:
      path=f'{tmp}/{i}.pt';subprocess.run([sys.executable,__file__,'--child',arm,str(seed),device,path],check=True);outputs.append(torch.load(path,weights_only=True))
     assert outputs[0]['loss']==outputs[1]['loss'] and all(torch.equal(t,outputs[1]['model'][k]) for k,t in outputs[0]['model'].items());rows.append(dict(arm=arm,seed=seed,device=device,exact=True))
 save(R/'resume-audit.json',dict(passed=True,rows=rows,note='Two fresh processes restore Adam/RNG then take same heldout synthetic next step; same-device equality only; exports unchanged.'))
if __name__=='__main__':
 if len(sys.argv)>1:child(sys.argv[2],int(sys.argv[3]),sys.argv[4],sys.argv[5])
 else:main()
