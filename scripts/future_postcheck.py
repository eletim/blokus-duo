from future_common import *
import subprocess,sys,traceback,fcntl

def main():
 lock=(R/'postcheck.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.monotonic()
 while not (R/'pipeline-done.json').exists():
  if (R/'failure.json').exists():raise RuntimeError('pipeline failure; inspect before further action')
  if time.monotonic()-start>6*3600:raise RuntimeError('six hour budget reached')
  time.sleep(15)
 for name,cmd in [('audit',[sys.executable,'scripts/future_audit.py']),('resume',[sys.executable,'scripts/future_resume.py']),('tests',[sys.executable,'-m','pytest','-q','tests/test_future_occupancy.py','tests/test_gen4.py'])]:
  if (R/f'post-{name}.json').exists():continue
  t=time.perf_counter()
  with (R/f'post-{name}.log').open('w') as f:subprocess.run(cmd,check=True,stdout=f,stderr=subprocess.STDOUT)
  save(R/f'post-{name}.json',dict(passed=True,seconds=time.perf_counter()-t))
 save(R/'postcheck-done.json',dict(passed=True))
if __name__=='__main__':
 try:main()
 except Exception:save(R/'postcheck-failure.json',dict(traceback=traceback.format_exc()));raise
