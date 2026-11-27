"""Post-hoc checkpoint diagnostics; never train or select models."""
from future_common import *
from future_train import tensors,metrics
from blokus.future_occupancy import FutureValue
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 assert (R/'evaluation-done.json').exists()
 torch.set_num_threads(1)
 d=tensors('train','cuda');sel=json.loads((R/'selection.json').read_text());out={};initial=[]
 for seed in SEEDS:
  _,source=load(seed,'native')
  for arm in 'ABC':
   name=f'{arm}-{seed}';m=FutureValue(source.model,arm).cuda();rows=[]
   for epoch in range(9):
    ck=torch.load(R/'models'/name/f'epoch-{epoch}.pt',map_location='cpu',weights_only=False);m.load_state_dict(ck['model'])
    rows.append(dict(epoch=epoch,train=metrics(m,d),dev=sel['models'][name]['curve'][epoch]['dev']))
   out[name]=rows;print('DIAGNOSTIC',name,flush=True)
  ck=torch.load(R/'models'/f'C-{seed}'/'epoch-0.pt',map_location='cpu',weights_only=False)
  equal=all(torch.equal(t,ck['model'][k]) for k,t in source.model.state_dict().items());zero=bool((ck['model']['map_value.weight']==0).all());assert equal and zero
  c=FutureValue(source.model,'C').eval();c.load_state_dict(ck['model']);a=FutureValue(source.model,'A').eval();a.load_state_dict(ck['model'])
  with torch.inference_mode():
   args=[d[k][:128].cpu().float() for k in ['board','state','extra']];cv=c(*args)[0];av=a(*args)[0];assert torch.equal(cv,av)
  exp=torch.load(R/f'C-{seed}.pt',map_location='cpu',weights_only=False)
  initial.append(dict(seed=seed,shared_gen4_tensors_equal=equal,map_projection_zero=zero,C_A_output_bit_equal_128=True,selected_epoch=exp['epoch'],selected_map_norm=float(exp['model']['map_value.weight'].norm())))
 save(R/'diagnostic-curves.json',dict(provenance='train reconstructed post hoc from saved epochs; dev original selection logs; no optimizer updates or reselection',curves=out,initial=initial))
 fig,axs=plt.subplots(3,3,figsize=(13,10))
 for i,arm in enumerate('ABC'):
  for seed in SEEDS:
   rows=out[f'{arm}-{seed}']
   for split,style in [('train','-'),('dev','--')]:
    axs[i,0].plot(range(9),[r[split]['value_mse'] for r in rows],style,label=f'{seed} {split}')
    if arm!='A':
     for j,metric in [(1,'ce'),(2,'brier')]:axs[i,j].plot(range(9),[r[split]['empty_cells']['-1:-1'][metric] for r in rows],style,label=f'{seed} {split}')
  for j,title in enumerate(['Value MSE','Empty-cell CE','Empty-cell Brier']):axs[i,j].set(title=f'{arm}: {title}',xlabel='epoch')
  axs[i,0].legend(fontsize=6)
 fig.tight_layout();fig.savefig(R/'diagnostic-curves.png',dpi=150);plt.close(fig)
if __name__=='__main__':main()
