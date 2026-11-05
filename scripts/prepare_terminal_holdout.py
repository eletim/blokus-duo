"""Reject training/used prefix or training-state collisions before evaluating fixed models."""
import sys,json,gzip,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from blokus.terminal_ppo import ROOT
from blokus.tournament import make_prefixes,digest,prefix_state
from blokus.visited_data import state_hash
from blokus.value_experiment import write
root=ROOT/'extended'
if (root/'holdout-disjoint-prefixes.json').exists():
 print('Disjoint fixture already frozen; leaving it unchanged.');raise SystemExit(0)
trainp=set();states=set()
for d in root.glob('seed-*'):
 for g in json.load(gzip.open(d/'games.json.gz','rt')):
  trainp.add(digest(g['prefix']));states.update(r['state_hash'] for r in g['targets'])
old=json.loads((root/'holdout-prefixes.json').read_text());overlap=[p['id'] for p in old['prefixes'] if p['id'] in trainp];start_overlap=[p['id'] for p in old['prefixes'] if state_hash(prefix_state(p)) in states]
write(root/'holdout-quality-failure.json',dict(original_seed=1480000,prefix_overlap=overlap,start_state_overlap=start_overlap,old_results_not_independent=True,new_seed=1500000,checkpoint_reselection=False,reason='New RNG seeds did not guarantee unique short prefixes; strict audit caught collision after arithmetic audit.'))
used=set(trainp)
for fixture in [old,json.loads((root/'validation-prefixes.json').read_text()),json.loads((ROOT/'validation-prefixes.json').read_text()),make_prefixes(1490000,16)]:used.update(p['id'] for p in fixture['prefixes'])
candidates=make_prefixes(1500000,256);selected=[];rejected=[]
for p in candidates['prefixes']:
 if p['id'] in used or state_hash(prefix_state(p)) in states:rejected.append(p['id']);continue
 selected.append(p)
 if len(selected)==96:break
assert len(selected)==96
new=dict(format=1,seed=1500000,count=96,sha256=digest(selected),prefixes=selected)
write(root/'holdout-disjoint-prefixes.json',new);write(root/'holdout-disjoint-plan.json',dict(seed=1500000,count=96,sha256=new['sha256'],rejected_before_96=rejected,selection_file_sha256=__import__('hashlib').sha256((root/'selection.json').read_bytes()).hexdigest(),filter='Exclude all training prefix hashes, training record state hashes at start, old validation/holdout/diagnostic prefix hashes; first 96 accepted of fixed seed stream; no game scores used',checkpoint_reselection=False))
archive=root/'holdout-1480000-overlap';archive.mkdir(exist_ok=True)
for path in list(root.glob('holdout*')):
 if path.is_file() and path.name not in ['holdout-quality-failure.json','holdout-disjoint-prefixes.json','holdout-disjoint-plan.json']:shutil.move(str(path),archive/path.name)
for d in root.glob('seed-*'):
 for path in d.glob('holdout-*.json'):
  dest=archive/d.name;dest.mkdir(exist_ok=True);shutil.move(str(path),dest/path.name)
print(dict(original_prefix_collisions=len(overlap),original_start_collisions=len(start_overlap),new_sha256=new['sha256'],rejected=len(rejected)))
