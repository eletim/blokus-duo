"""CPU8 control with only the generation encoding/copy optimization enabled."""
import argparse
from pathlib import Path
import gpu_generation as ref
from blokus.ppo_policy import PPOAgent as OriginalAgent
original_worker=ref.worker

def worker(*args):
 from blokus.generation_encoding import encode,observe,child
 import blokus.ppo_search_deeper as search
 import blokus.ppo_value_leaf as leaf
 search.child=child;leaf.value_inputs=encode
 def agent(*args,**kwargs):
  p=OriginalAgent(*args,**kwargs);p._observation=lambda s:observe(s,p.channels);return p
 ref.PPOAgent=agent
 return original_worker(*args)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--repeat',type=int,default=1);a=p.parse_args();ref.R=Path(a.root);ref.worker=worker;ref.benchmark('cpu-fast',a.repeat)
