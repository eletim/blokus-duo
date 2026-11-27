from future_common import *
from future_generate import generate
from future_data import prepare
from future_train import train
from future_eval import bank,evaluate
import traceback,fcntl

def main():
 lock=(R/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 plan=json.loads((R/'FINAL_PLAN.json').read_text())
 for name,h in plan['code_sha'].items():assert sha(name)==h,name
 for phase,n in plan['games'].items():
  if not (R/f'{phase}-generation.json').exists():generate(phase,n)
 prepare();bank();train()
 files=[R/'FINAL_PLAN.json',R/'selection.json',R/'eval-bank.json']+list(R.glob('[ABC]-*.pt'))
 save(R/'evaluation-manifest.json',{str(p):sha(p) for p in files})
 evaluate();save(R/'pipeline-done.json',dict(computed=True,audit_pending=True))
if __name__=='__main__':
 try:main()
 except Exception:save(R/'failure.json',dict(traceback=traceback.format_exc()));raise
