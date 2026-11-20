"""Focused driver regressions: identical update math and no strength-based model choice."""
import ast
from pathlib import Path

def test_training_algorithm_preserved():
 old=ast.parse(Path('scripts/placement_large_train.py').read_text());new=ast.parse(Path('scripts/placement_extension_train.py').read_text())
 for name in ('step','rng_state','restore_rng','atomic_save','arrays'):
  a=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name);b=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name==name)
  assert ast.dump(a)==ast.dump(b)
 # The entire trainer differs only in declared stage game-count metadata and root path.
 a=Path('scripts/placement_large_train.py').read_text();b=Path('scripts/placement_extension_train.py').read_text()
 a=a.replace('Real, resumable CUDA value learning on matched 24,576-game A/C data.','Same verified updates, parameterized only by fixed cumulative data stage.').replace("R=Path('artifacts/value-placement-24576')","R=Path(os.environ['PLACEMENT_STAGE']);GAMES=int(R.name[1:])").replace('train_games=24576','train_games=GAMES')
 assert a==b

def test_stage_composition_and_fixed_opponents():
 import sys
 sys.path.insert(0,'scripts')
 import placement_extension_eval as q
 assert str(q.source('P',36864,1751401)).endswith('24576/A-1751401.pt')
 assert str(q.source('P',49152,1751401)).endswith('n36864/A-1751401.pt')
 assert str(q.source('R',49152,1751401)).endswith('24576/A-1751401.pt')
 assert str(q.source('Q',49152,1751401)).endswith('n3072/A-1751401.pt')
 assert str(q.source('A',49152,1751401)).endswith('n49152/A-1751401.pt')
 assert str(q.source('C',49152,1751401)).endswith('n49152/C-1751401.pt')
 #256-game shards divide both per-seed stages exactly, retaining paired labels.
 for count,limit in [(36864,4096),(49152,8192)]:
  shards=[range(i,i+256) for i in range(0,8192,256)]
  chosen=[s for s in shards if all(g<limit for g in s)]
  assert sum(len(s) for s in chosen)*3+24576==count
