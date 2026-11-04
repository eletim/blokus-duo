"""Fixed checkpoints/alphas, repaired shared scheduler, fresh 96-prefix holdout."""
import json
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from blokus.ai import seed_all
from blokus.research import source_hashes,sha
from blokus.value_experiment import write,evaluate
from blokus.tournament import make_prefixes

root=Path('artifacts/paired-ranking');out=root/'confirmed';out.mkdir(parents=True,exist_ok=True)
if (out/'config.json').exists():raise ValueError('existing confirmation; preserve it')
seed_all(301);start=time.perf_counter()
specs=json.loads((root/'play-specs.json').read_text())
write(out/'config.json',dict(source_hashes=source_hashes(),holdout_seed=230000,prefixes=96,budget_ms=6,
    selection_sha256=sha(root/'selection.json'),specs_sha256=sha(root/'play-specs.json'),
    reason='repair search starvation after one wall-time spike; same frozen checkpoints and alphas; no reselection'))
prefixes=make_prefixes(230000,96);write(out/'holdout-prefixes.json',prefixes)
pairs=[(f'ranking-{s}',other) for s in [301,302,303] for other in ['static','prior']]+[('prior','static')]
evaluate(specs,pairs,prefixes,out/'holdout-results.json')
write(out/'summary.json',dict(completed=True,wall_seconds=time.perf_counter()-start,new_training_seconds=0,games=96*2*7))
print('CONFIRM COMPLETE',flush=True)
