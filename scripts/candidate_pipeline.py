"""Resumable finite orchestration; inspect failure.json rather than blindly restart."""
import time,json,traceback
from pathlib import Path
from candidate_data import R,generate,diagnose,save
from candidate_train import train
from candidate_eval import evaluate

def main():
 # A separately started bank preparation owns all writes until its manifest exists.
 start=time.perf_counter()
 while not (R/'bank-manifest.json').exists():
  if time.perf_counter()-start>1200:raise RuntimeError('bank preparation exceeded20min; inspect existing process')
  time.sleep(10)
 generate('diagnosis')
 if not diagnose():save(R/'pipeline-result.json',dict(state='diagnostic-rejected',needs_consultation=True));return
 generate('train');generate('dev');train()
 passed=evaluate('validation')
 if passed:evaluate('confirmation')
 save(R/'pipeline-result.json',dict(state='computed',confirmation=passed,seconds=time.perf_counter()-start,audit_pending=True))
if __name__=='__main__':
 try:main()
 except Exception:
  save(R/'failure.json',dict(traceback=traceback.format_exc()));raise
