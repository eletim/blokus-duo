"""Fresh process interrupted-game restart; no mid-node checkpoint claim."""
import gzip,json,shutil,subprocess,sys
from pathlib import Path
import frontier_generation as f
R=f.R;src=R/'parallel-1';dst=R/'parallel-3';assert not dst.exists();shutil.copytree(src,dst)
files=sorted(dst.glob('*.gz'));lost=files[0];expected=json.load(gzip.open(lost,'rt'));lost.unlink();lost.with_suffix('.tmp').write_text('incomplete never a label')
prior={p.name:f.ref.e.sha(p) for p in files[1:]}
subprocess.run([sys.executable,'scripts/frontier_parallel.py','--repeat','3'],check=True,stdout=subprocess.DEVNULL)
actual=json.load(gzip.open(lost,'rt'));assert expected['task']==actual['task'];assert actual['moves']==expected['moves'] and actual['scores']==expected['scores']
assert all(f.ref.e.sha(dst/name)==h for name,h in prior.items());assert not lost.with_suffix('.tmp').exists()
f.ref.write(R/'resume-check.json',dict(passed=True,unchanged_files=len(prior),restarted_games=1,scope='game-boundary restart from saved initial RNG, not suspended stack serialization',regenerated_sha=f.ref.e.sha(lost)))
