from future_common import *
from blokus.future_occupancy import FutureValue
import torch.nn.functional as F

def tensors(phase,device):
 a=np.load(R/f'{phase}-data.npz');return {k:torch.as_tensor(a[k],device=device) for k in a.files}
def loss(model,d,ix):
 v,l,p=model(d['board'][ix],d['state'][ix].float(),d['extra'][ix].float(),with_map=model.arm!='A')
 w=d['weight'][ix].float();err=(v-d['value'][ix])**2;vl=(err*w).sum()/w.sum();ol=vl*0
 if model.arm!='A':
  ce=F.cross_entropy(l,d['occupancy'][ix].long(),reduction='none');mask=d['empty'][ix];ce=(ce*mask).flatten(1).sum(1)/mask.flatten(1).sum(1).clamp_min(1);ol=(ce*w).sum()/w.sum()
 return vl+.25*ol,vl,ol

def metrics(model,d):
 model.eval();n=len(d['value']);sums={};cells={};values=[]
 with torch.inference_mode():
  for start in range(0,n,256):
   ix=slice(start,start+256);v,l,p=model(d['board'][ix],d['state'][ix].float(),d['extra'][ix].float(),with_map=model.arm!='A');values.append(v.cpu());w=d['weight'][ix].float()
   sums['value_sum']=sums.get('value_sum',0)+float((((v-d['value'][ix])**2)*w).sum());sums['weight']=sums.get('weight',0)+float(w.sum())
   if model.arm!='A':
    y=d['occupancy'][ix].long();mask=d['empty'][ix];ce=F.cross_entropy(l,y,reduction='none');oh=F.one_hot(y,3).permute(0,3,1,2);br=((p-oh)**2).sum(1)
    for stage in [-1,0,1,2]:
     m=mask if stage==-1 else mask&(d['stage'][ix]==stage)[:,None,None]
     for cls in [-1,0,1,2]:
      z=m if cls==-1 else m&(y==cls);key=f'{stage}:{cls}';q=cells.setdefault(key,dict(n=0,ce=0.,brier=0.,accuracy=0.));q['n']+=int(z.sum());q['ce']+=float(ce[z].sum());q['brier']+=float(br[z].sum());q['accuracy']+=float((p.argmax(1)[z]==y[z]).sum())
 for q in cells.values():
  for k in ['ce','brier','accuracy']:q[k]/=max(q['n'],1)
 return dict(value_mse=sums['value_sum']/sums['weight'],empty_cells=cells)

def train():
 if (R/'selection.json').exists():return
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);d=tensors('train','cuda');dev=tensors('dev','cuda');selection={};start=time.perf_counter()
 for seed in SEEDS:
  _,source=load(seed,'native')
  for arm in 'ABC':
   torch.manual_seed(seed+924);model=FutureValue(source.model,arm).cuda();opt=torch.optim.Adam(model.parameters(),lr=1e-4);gen=torch.Generator().manual_seed(seed+924);curve=[];updates=0;dest=R/'models'/f'{arm}-{seed}';dest.mkdir(parents=True,exist_ok=True)
   initial={k:t.detach().cpu().clone() for k,t in model.state_dict().items()}
   for epoch in range(9):
    if epoch:
     model.train();order=torch.randperm(len(d['value']),generator=gen)
     for ix in order.split(128):
      ix=ix.cuda();opt.zero_grad();total,vl,ol=loss(model,d,ix);assert torch.isfinite(total);total.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();updates+=1
    mt=metrics(model,dev);curve.append(dict(epoch=epoch,updates=updates,dev=mt))
    torch.save(dict(model={k:t.detach().cpu() for k,t in model.state_dict().items()},optimizer=opt.state_dict(),shuffle_rng=gen.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),epoch=epoch,updates=updates,arm=arm,seed=seed),dest/f'epoch-{epoch}.pt')
    print('TRAIN',arm,seed,epoch,updates,mt['value_mse'],flush=True)
   chosen=min(range(9),key=lambda i:curve[i]['dev']['value_mse']);ck=torch.load(dest/f'epoch-{chosen}.pt',map_location='cpu',weights_only=False);changed=[k for k,t in ck['model'].items() if not torch.equal(t,initial[k])]
   torch.save(dict(model=ck['model'],arm=arm,seed=seed,epoch=chosen,updates=ck['updates'],format='future-occupancy-v1',gen4_manifest_sha=sha('models/gen4/manifest.json')),R/f'{arm}-{seed}.pt')
   selection[f'{arm}-{seed}']=dict(epoch=chosen,dev=curve[chosen]['dev'],total_updates=updates,changed=changed,parameters=sum(p.numel() for p in model.parameters()),curve=curve)
 save(R/'selection.json',dict(rule='per-arm/seed minimum dev value MSE; earliest tie; epoch0 allowed',models=selection,seconds=time.perf_counter()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated()))
if __name__=='__main__':train()
