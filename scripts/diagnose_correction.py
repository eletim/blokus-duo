"""Post-selection diagnostic only; never used to retune checkpoints."""
import sys,json,time,gzip,random
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from blokus.correction_data import ROOT
from blokus.correction import CorrectionNet,CorrectionAgent
from blokus.aligned_data import BASE
from blokus.engine import State
from blokus.research import clone
from blokus.reply_search import reset_game
from blokus.correction_train import metrics
from blokus.value_experiment import write
from blokus.ai import seed_all

def main():
 seed_all(1000);sel=json.loads((ROOT/'selection.json').read_text());raw=np.load(ROOT/'data.npz');data={k:torch.as_tensor(raw[k],device='cuda') for k in raw.files};out=[]
 for seed,ck in sel['checkpoints'].items():
  model=CorrectionNet().cuda();model.load_state_dict(torch.load(ck['path'],map_location='cpu',weights_only=False)['model']);model.eval()
  report={str(split):metrics(model,data,split,sel['choice']['alpha']) for split in [0,1,2]}
  with torch.no_grad():
   ids=torch.where(data['split']==1)[0];pred=model(data['board'][ids],data['state'][ids],data['x'][ids]);base=data['initial'][ids];values=(base+sel['choice']['alpha']*pred).nan_to_num(nan=-float('inf'));chosen=values.argmax(1);known=data['mask'][ids].gather(1,chosen[:,None]).squeeze(1)
  out.append(dict(seed=seed,splits=report,full_candidate_choice_labeled_fraction=float(known.float().mean())))
 mask=raw['mask'];target=raw['target'];groups={}
 for name,m in [('initial_top4',mask&~raw['random']),('uniform_other',mask&raw['random'])]:
  groups[name]=dict(labels=int(m.sum()),mean_delta=float(target[m].mean()),std_delta=float(target[m].std()),mean_absolute_delta=float(abs(target[m]).mean()))
 records=json.load(gzip.open(ROOT/'teacher-labels.json.gz','rt'));coverage=[]
 for seed,ck in sel['checkpoints'].items():
  bot=CorrectionAgent(BASE,sel['choice']['budget'],ck['path'],alpha=sel['choice']['alpha']);rows=[]
  for r in records:
   if r['split']!=1:continue
   reset_game(bot);s=clone(State(**r['state']));bot.act(s,random.Random(1060000+r['game']))
   truth=np.array([v if v is not None else np.nan for v in r['final']]);best=np.nanmax(truth);optimal=np.flatnonzero(truth>=best-4e-6);known=bot.last_known;choice=int(bot.final_values.argmax());base_choice=int(np.nanargmax(bot.initial_values)) if known.any() else 0
   rows.append(dict(game=r['game'],teacher_optimum_covered=bool(known[optimal].any()),choice_labeled=bool(np.isfinite(truth[choice])),teacher_retained=bool(choice in optimal),zero_teacher_retained=bool(base_choice in optimal),completed=int(known.sum()),candidates=len(known),ms=bot.decisions[-1]['ms']))
  coverage.append(dict(seed=seed,rows=rows))
 write(ROOT/'diagnostics.json',dict(selected_models=out,label_groups=groups,decision_coverage=coverage,note='same-depth candidate-set regret; unknown candidates excluded from reference. Full-set choice coverage reports extrapolation. Post-selection diagnostic, not retuning.'))
if __name__=='__main__':main()
