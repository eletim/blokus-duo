"""End-to-end exact continuation, including selfplay action sampling and Adam."""
from argparse import Namespace
import torch
from blokus.ai import Net,seed_all,load
from blokus.__main__ import train

def test_training_resume_equals_uninterrupted(tmp_path):
    def run(name,games,resume=None):
        seed_all(91);model=Net()
        train(Namespace(output=str(tmp_path/name),lr=.001,resume=resume,games=games,seconds=30,imitation_games=1,device='cpu'),model)
        return tmp_path/name/'latest.pt'
    full=run('full',4)
    part=run('part',2)
    resumed=run('resumed',2,str(part))
    a=torch.load(full,weights_only=False);b=torch.load(resumed,weights_only=False)
    assert a['episode']==b['episode']==4
    assert a['phase']==b['phase']=='selfplay'
    assert all(torch.equal(v,b['model'][k]) for k,v in a['model'].items())
    for k,st in a['optimizer']['state'].items():
        assert all(torch.equal(v,b['optimizer']['state'][k][n]) for n,v in st.items())
