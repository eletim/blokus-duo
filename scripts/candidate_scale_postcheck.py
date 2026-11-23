"""Bounded post-evaluation checks, sequential and restart-safe."""
import json,subprocess,sys,time,traceback
from candidate_scale_data import R,save

def main():
 start=time.monotonic()
 while not (R/'pipeline-result.json').exists():
  if (R/'failure.json').exists():raise RuntimeError('pipeline failed; inspect before resuming')
  if time.monotonic()-start>7*3600:raise RuntimeError('wait budget exhausted; inspect existing pipeline')
  time.sleep(10)
 assert json.loads((R/'pipeline-result.json').read_text())['computed']
 commands=[('audit',[sys.executable,'scripts/candidate_scale_audit.py'])]
 for n in [3072,12288]:
  commands.append((f'resume-{n}',[sys.executable,'scripts/candidate_scale_resume.py','--stage',str(n)]))
  code=f"import candidate_train as t; import candidate_dev_audit as d; from candidate_scale_data import R; t.R=d.R=R/'n{n}'; d.main()"
  commands.append((f'dev-{n}',[sys.executable,'-c',code]))
  commands.append((f'cli-{n}',[sys.executable,'scripts/candidate_play.py','--policy',str(R/f'n{n}/policy-1751401.pt')]))
 commands.append(('tests',[sys.executable,'-m','pytest','-q','tests/test_candidate_policy.py','tests/test_gen4.py']))
 for name,cmd in commands:
  if (R/f'post-{name}.ok.json').exists():continue
  t=time.perf_counter()
  with (R/f'post-{name}.log').open('w') as log:subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT)
  save(R/f'post-{name}.ok.json',dict(seconds=time.perf_counter()-t,command=cmd))
 save(R/'postchecks-result.json',dict(passed=True,seconds_including_wait=time.monotonic()-start))
if __name__=='__main__':
 try:main()
 except Exception:save(R/'postchecks-failure.json',dict(traceback=traceback.format_exc()));raise
