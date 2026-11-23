"""Sequential post-evaluation checks; never overlaps the timed comparison."""
import os,time,json,subprocess,sys,traceback
from pathlib import Path
from candidate_data import R,save

def main():
 start=time.monotonic()
 while not (R/'pipeline-result.json').exists():
  if (R/'failure.json').exists():raise RuntimeError('pipeline failed; inspect failure.json')
  if time.monotonic()-start>5400:raise RuntimeError('timed comparison wait limit exceeded; inspect before restart')
  time.sleep(10)
 assert json.loads((R/'pipeline-result.json').read_text())['state']=='computed'
 commands=[('audit',[sys.executable,'scripts/candidate_audit.py']),('resume',[sys.executable,'scripts/candidate_resume_check.py']),('dev-audit',[sys.executable,'scripts/candidate_dev_audit.py']),('tests',[sys.executable,'-m','pytest','-q','tests/test_candidate_policy.py','tests/test_gen4.py'])]
 for name,cmd in commands:
  if (R/f'post-{name}.ok.json').exists():continue
  t=time.perf_counter()
  with (R/f'post-{name}.log').open('w') as log:subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT)
  save(R/f'post-{name}.ok.json',dict(seconds=time.perf_counter()-t,command=cmd))
 cmd=[sys.executable,'scripts/candidate_play.py','--policy',str(R/'policy-1751401.pt')]
 if not (R/'post-cli.ok.json').exists():
  t=time.perf_counter()
  with (R/'post-cli.log').open('w') as log:subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT)
  save(R/'post-cli.ok.json',dict(seconds=time.perf_counter()-t,command=cmd))
 save(R/'postchecks-result.json',dict(passed=True,seconds=time.monotonic()-start))
if __name__=='__main__':
 try:main()
 except Exception:save(R/'postchecks-failure.json',dict(traceback=traceback.format_exc()));raise
