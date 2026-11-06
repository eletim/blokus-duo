import hashlib,json,random
from pathlib import Path
import pytest,torch
from blokus.ppo_policy import PPOAgent,TerminalNet,collate,resolve_seed
from blokus.engine import State,play
ROOT=Path(__file__).resolve().parents[1]

def test_generation_seed_validation():
 assert resolve_seed('gen3')==1751401 and resolve_seed('gen3',1401)==1751401
 assert resolve_seed('gen2')==1401
 with pytest.raises(ValueError,match='gen2 requires'):resolve_seed('gen2',1751401)
 with pytest.raises(ValueError,match='gen3 requires'):resolve_seed('gen3',123)

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_gen3_frozen_exports_and_outputs(device):
 if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
 torch.set_num_threads(1)
 for seed,row in json.loads((ROOT/'artifacts/ppo-gen3/manifest.json').read_text()).items():
  path=ROOT/row['public_path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
  ck=torch.load(path,weights_only=True,map_location='cpu');reference=TerminalNet(4).to(device);reference.load_state_dict(ck['model'])
  a=PPOAgent(int(seed),device,generation='gen3');assert a.channels==4 and a.metadata['iteration']==5120
  s=State()
  for i in range(6):
   obs,qs=a._observation(s);batch=collate([obs],device)
   with torch.no_grad():x=reference(*batch);y=a.model(*batch)
   assert all(torch.equal(u,v) for u,v in zip(x,y))
   scores=x[0][0].cpu().numpy();import numpy as np
   index=random.Random(i).choice(np.flatnonzero(scores>=scores.max()-1e-7));expected=qs[index].move if qs else None
   assert a.act(s,random.Random(i))==expected;play(s,expected)
