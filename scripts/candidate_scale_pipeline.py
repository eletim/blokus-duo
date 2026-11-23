"""Finite serial stages; preserves completed work on process restart."""
import time,json,traceback
import candidate_train as train_module
from candidate_scale_data import R,prepare,generate,save
from candidate_scale_eval import evaluate,finish_summary

def main():
 start=time.perf_counter();prepare()
 for n in [3072,12288]:
  stage=R/f'n{n}'
  if not (stage/'generation.json').exists():generate(n)
  if not (stage/'training-time.json').exists():train_module.R=stage;train_module.train()
 for n in [3072,12288]:
  for c in ['Frozen','old768']:
   if not (R/f'n{n}'/f'vs-{c}-summary.json').exists():evaluate(n,c)
 finish_summary();save(R/'pipeline-result.json',dict(computed=True,seconds=time.perf_counter()-start,audit_pending=True))
if __name__=='__main__':
 try:main()
 except Exception:save(R/'failure.json',dict(traceback=traceback.format_exc()));raise
