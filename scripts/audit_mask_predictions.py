"""Independent NumPy decision/MSE arithmetic for both retained CUDA checkpoints."""
import sys,json
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from blokus.mask_model import CandidateNet
from blokus.mask_data import ROOT
from blokus.ai import seed_all
from blokus.value_experiment import write
seed_all(1100);raw=np.load(ROOT/'compact.npz');d={k:raw[k] for k in raw.files};ix=np.flatnonzero(d['bank']&(d['fold']==0));checks={}
for rep in ['summary','cells']:
 model=CandidateNet(rep).cuda();model.load_state_dict(torch.load(ROOT/f'n2400-f0-s1101-{rep}/step-1000.pt',weights_only=False)['model']);model.eval();games={}
 with torch.no_grad():
  for block in np.array_split(ix,8):
   pred=model(*[torch.as_tensor(d[k][block],device='cuda') for k in ['board','state','x']]).cpu().numpy()
   for j,row in enumerate(block):
    mask=d['mask'][row];p=pred[j,mask];t=d['target'][row,mask];base=d['initial'][row,mask];truth=base+t
    v=[np.mean((p-t)**2),np.mean(t*t),truth.max()-truth[np.argmax(base+.5*p)],truth.max()-truth[np.argmax(base)]];games.setdefault(str(d['game'][row]),[]).append(v)
 ledger=json.loads((ROOT/f'n2400-f0-s1101-{rep}/result.json').read_text())['logs'][0]['test_games'];assert set(games)==set(ledger)
 for g,v in games.items():np.testing.assert_allclose(np.mean(v,0),ledger[g],atol=2e-5,rtol=1e-5)
 checks[rep]=dict(games=len(games),roots=len(ix),pass_=True)
write(ROOT/'prediction-audit.json',dict(checks=checks,note='Independent NumPy reductions on saved checkpoint predictions; floating arithmetic tolerance 2e-5 absolute.'))
