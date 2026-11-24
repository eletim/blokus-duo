from future_common import *
from blokus.future_occupancy import FutureValue,terminal_labels
from blokus.value_inputs import encode
from blokus.placement_value import mobility
from audit_placement import replay
from future_train import loss,metrics

def main():
 torch.set_num_threads(1);gs=[json.loads(p.read_text()) for p in sorted((R/'games/pilot').glob('*.json'))];assert len(gs)==12;rows=[]
 for g in gs:
  assert replay(g['moves'])==g['scores'];final=state(g['moves']);assert final.boards==g['boards'] and final.done
  for ply in [g['prefix_length'],20,28]:
   if ply>=len(g['moves'])-2:continue
   s=state(g['moves'][:ply]);b,st=encode(s);y=terminal_labels(final.boards,s.turn);mask=~b[:2].any(0)
   assert np.all(y[b[0].astype(bool)]==0) and np.all(y[b[1].astype(bool)]==1)
   rows.append((b[:4],st,mobility(s),y,mask,float(np.sign(g['scores'][s.turn]-g['scores'][1-s.turn])),1.,min(ply//12,2)))
 keys=['board','state','extra','occupancy','empty','value','weight','stage'];d={k:torch.as_tensor(np.stack([a[i] for a in rows]),device='cuda') for i,k in enumerate(keys)};_,src=load(SEEDS[0],'native');out={}
 for arm in 'ABC':
  torch.manual_seed(924);m=FutureValue(src.model,arm).cuda();opt=torch.optim.Adam(m.parameters(),lr=1e-4);before={k:t.detach().clone() for k,t in m.state_dict().items()};start=time.perf_counter()
  for i in range(8):
   opt.zero_grad();l,_,_=loss(m,d,slice(None));assert torch.isfinite(l);l.backward();opt.step()
  torch.cuda.synchronize();secs=time.perf_counter()-start;m.cpu().eval();b,st,ex=[d[k][:1].cpu() for k in ['board','state','extra']]
  with torch.inference_mode():
   start=time.perf_counter()
   for i in range(200):m(b,st.float(),ex.float(),with_map=False)
  out[arm]=dict(parameters=sum(p.numel() for p in m.parameters()),updates=8,changed=sum(not torch.equal(t.cpu(),before[k].cpu()) for k,t in m.state_dict().items()),cuda_seconds=secs,inference_ms=(time.perf_counter()-start)*5)
 save(R/'pilot-check.json',dict(passed=True,games=12,rows=len(rows),arms=out,peak_cuda_bytes=torch.cuda.max_memory_allocated(),generation=json.loads((R/'pilot-generation.json').read_text())))
if __name__=='__main__':main()
