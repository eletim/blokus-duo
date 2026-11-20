"""Reuse verified independent replay/encoding; compose two cumulative, matched stages."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import concurrent.futures,json,time,collections
from pathlib import Path
import placement_large_features as f
R=Path('artifacts/value-placement-49152');PREV=Path('artifacts/value-placement-24576');f.R=R

def main():
 f.e.setup();start=time.perf_counter();root=R/'feature-shards';root.mkdir(exist_ok=True);bank=json.loads((R/'bank.json').read_text());jobs=[]
 for phase in ('train','dev'):
  for seed in f.e.SEEDS:
   tasks=[t for t in bank['games'] if t['phase']==phase and t['seed']==seed]
   for i in range(0,len(tasks),256):jobs.append((len(jobs),tasks[i:i+256]))
 with concurrent.futures.ProcessPoolExecutor(max_workers=8,initializer=f.init) as pool:results=list(pool.map(f.worker,jobs))
 sets={'train':set(),'dev':set()};counts=collections.Counter();allgames=set()
 for job,d in enumerate(results):
  labels=f.readgz(root/f'{job:03d}.labels.json.gz');sets[d['phase']].update(labels['visited']);counts[d['phase']]+=len(d['games'])
  for g in d['games']:
   key=(g['phase'],g['seed'],g['id']);assert key not in allgames;allgames.add(key)
 assert counts=={'train':24576,'dev':192} and not sets['train']&sets['dev']
 previous=json.loads((PREV/'data-manifest.json').read_text());assert json.loads((PREV/'audit.json').read_text())['pass_']
 for p,h in previous['file_shas'].items():assert f.e.sha(p)==h
 for phase,seen in sets.items():f.writegz(R/f'{phase}-visited.json.gz',dict(states=sorted(seen)))
 for count,limit in [(36864,4096),(49152,8192)]:
  dest=R/f'n{count}';dest.mkdir(exist_ok=True);(dest/'PLAN.md').write_bytes((R/'PLAN.md').read_bytes())
  chosen=[(j,d) for j,d in enumerate(results) if d['phase']=='dev' or all(g['id']<limit for g in d['games'])]
  trainfiles=[str(root/f'{j:03d}.npz') for j,d in chosen if d['phase']=='train'];devfiles=[str(root/f'{j:03d}.npz') for j,d in chosen if d['phase']=='dev']
  trainfiles=previous['train_files']+trainfiles
  manifest=dict(train_files=trainfiles,dev_files=devfiles,file_shas={p:f.e.sha(p) for p in trainfiles+devfiles},previous_manifest_sha=f.e.sha(PREV/'data-manifest.json'),previous_audit_sha=f.e.sha(PREV/'audit.json'),old_training_games=24576,new_training_games=count-24576,training_games=count,dev_games=192,old_labels=previous['old_labels']+previous['new_labels'],new_labels=sum(d['labels'] for j,d in chosen if d['phase']=='train'),dev_labels=sum(d['labels'] for j,d in chosen if d['phase']=='dev'),source_bank_sha=f.e.sha(R/'bank.json'),shards=[d for j,d in chosen],seconds=time.perf_counter()-start)
  assert sum(len(d['games']) for j,d in chosen if d['phase']=='train')==count-24576
  f.e.write(dest/'data-manifest.json',manifest)
 f.e.write(R/'data-audit.json',dict(pass_=True,counts=dict(counts),train_dev_overlap=0,prior_overlap=0,shards=results,seconds=time.perf_counter()-start))
 print('TWO DATA STAGES READY',flush=True)
if __name__=='__main__':main()
