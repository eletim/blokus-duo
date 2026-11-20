"""Interrupt only our own process group, retain completed games, resume remainder."""
import gzip,json,os,signal,subprocess,sys,time
from pathlib import Path
import frontier_generation as f
from audit_placement import replay
R=Path('artifacts/gpu-async512');out=R/'restart';assert not out.exists();bank=R/'quality/bank.json'
cmd=[sys.executable,'scripts/async_generation.py','--bank',str(bank),'--out',str(out),'--concurrency','512','--fast-encoding','--graphs']
with open(R/'restart-interrupted.log','w') as log:
 p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);deadline=time.monotonic()+90
 while not list(out.glob('[0-9]*.json.gz')) and p.poll() is None and time.monotonic()<deadline:time.sleep(.05)
 assert p.poll() is None,'generation exited before interruption'
 os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=15)
 # Give the explicitly owned children time to exit; never target other groups.
 time.sleep(.5)
files=list(out.glob('[0-9]*.json.gz'));assert 0<len(files)<24
before={str(path):f.ref.e.sha(path) for path in files};tasks=json.loads(bank.read_text())['games'];missing=[g for g in tasks if not (out/f"{g['seed']}-{g['id']}.json.gz").exists()]
partial=(out/f"{missing[0]['seed']}-{missing[0]['id']}.json.gz").with_suffix('.tmp');partial.write_text('interrupted incomplete payload')
with open(R/'restart-completed.log','w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
assert all(f.ref.e.sha(path)==sha for path,sha in before.items());equal=0
for task in tasks:
 name=f"{task['seed']}-{task['id']}.json.gz";d=json.load(gzip.open(out/name,'rt'));assert d['task']==task and replay(d['moves'])==d['scores'];assert all(s['forced_pass'] or s['depth']==6 for s in d['stats'])
 ref=json.load(gzip.open(R/'quality/cpu-1'/name,'rt'));equal+=d['moves']==ref['moves']
assert not partial.exists()
f.ref.write(R/'restart-check.json',dict(completed_retained=len(before),regenerated=len(missing),all_games=24,retained_sha_unchanged=True,trajectories_equal_cpu=equal,game_boundary_only=True,terminated_owned_group_pid=p.pid,config=json.loads((out/'run.json').read_text())))
