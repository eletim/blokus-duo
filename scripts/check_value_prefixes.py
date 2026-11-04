"""Verify train/evaluation prefix state separation, not just move history hashes."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from blokus.engine import State,Move,play
root=Path('artifacts/value-search')
def statekey(moves):
 s=State()
 for row in moves:play(s,Move(*row) if row else None)
 return (tuple(s.boards),tuple(s.remaining),tuple(s.last),s.turn,s.passes)
train=[]
for p in [root/'run/games.prefixes.json',root/'run/branches.prefixes.json',root/'compact/expanded.prefixes.json']:train+=json.loads(p.read_text())
a={statekey(x) for x in train};sets={};checks={}
for p in [root/'run/validation-prefixes.json',root/'run/holdout1-prefixes.json',root/'run/holdout2-prefixes.json',root/'compact/holdout-prefixes.json',root/'confirm/holdout-prefixes.json']:
 b={statekey(x['moves']) for x in json.loads(p.read_text())['prefixes']};sets[str(p)]=b;checks[str(p)]=len(a&b)
assert not any(checks.values())
for name,b in sets.items():
 for other,c in sets.items():
  if name!=other:assert not b&c
(root/'prefix-overlap.json').write_text(json.dumps(dict(train_prefixes=len(train),unique_train_states=len(a),train_eval_state_overlap=checks,eval_sets_mutually_disjoint=True),indent=2))
