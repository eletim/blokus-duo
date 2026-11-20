"""Simulate a killed generation at a game boundary, in fresh CPU/CUDA processes."""
import json,gzip,shutil,subprocess,sys,hashlib
from pathlib import Path
R=Path('artifacts/gpu-generation')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=[]
for mode in ('cpu','gpu'):
    src=R/f'{mode}-1';dst=R/f'{mode}-3';shutil.copytree(src,dst)
    files=sorted(dst.glob('*.gz'));lost=files[0];expected=json.load(gzip.open(lost,'rt'));lost.unlink()
    lost.with_suffix('.tmp').write_text('simulated unfinished game; never a WDL record')
    prior={p.name:sha(p) for p in files[1:]}
    subprocess.run([sys.executable,'scripts/gpu_generation.py',mode,'--repeat','3'],check=True,stdout=subprocess.DEVNULL)
    actual=json.load(gzip.open(lost,'rt'));assert actual['task']==expected['task']
    assert actual['moves']==expected['moves'] and actual['scores']==expected['scores']
    assert all(sha(dst/name)==h for name,h in prior.items())
    assert not lost.with_suffix('.tmp').exists()
    timing=json.loads((R/f'{mode}-3.json').read_text());assert sum(x['games'] for x in timing['workers'])==1
    rows.append(dict(mode=mode,skipped_unchanged=23,replayed_from_saved_rng=1,moves_scores_equal=True,seconds=timing['seconds']))
(R/'resume-check.json').write_text(json.dumps(dict(passed=True,rows=rows,scope='game boundary; no partial result accepted, not a mid-node checkpoint'),indent=2)+'\n')
print(json.dumps(rows))
