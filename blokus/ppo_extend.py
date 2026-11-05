"""Fixed follow-up learning-curve stages, separate from frozen confirmation."""
import argparse,json
from pathlib import Path
from . import ppo_curve_eval as q
from .value_experiment import write
from .research import sha
R=Path('artifacts/ppo-4096');OLD=Path('artifacts/ppo-learning-curve')
def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['audit_train','validation','holdout','audit_eval']);mode=p.parse_args().mode;q.R=R;q.OLD=OLD
 if mode=='audit_train':
  q.audit_train()
  from .tournament import read_prefixes,prefix_state
  from .visited_data import state_hash
  bank=read_prefixes(R/'confirmation-prefixes.json');prefixes=q.hashes('training-prefixes');states=q.hashes('training-states');overlap=prefixes&{r['id'] for r in bank['prefixes']};assert not overlap
  write(R/'confirmation-training-exclusion.json',dict(pass_=True,prefix_overlap=0,all_visited_overlap=len(states&{state_hash(prefix_state(r)) for r in bank['prefixes']}),note='Frozen confirmation was not used for learning; prefix generation rejects its states. Incidental future game-state overlap is separately audited.'))
 elif mode=='validation':q.validation(1590000)
 elif mode=='holdout':q.holdout(1600000,('previous','oldvalue'))
 else:q.audit_eval(('previous','oldvalue'))
if __name__=='__main__':main()
