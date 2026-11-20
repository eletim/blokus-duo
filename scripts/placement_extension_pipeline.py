"""Resumable, finite five-hour extension; phases never overlap heavy work."""
import os,sys,time,json,subprocess,urllib.request,uuid,datetime,threading
from pathlib import Path
R=Path('artifacts/value-placement-49152');PY=sys.executable
BASE=dict(humanRequestId='792384ab-c691-4f40-8981-ad987a8bc3c4',workId='3a6b43dc-82be-4152-a19c-fedf5aa5c68e',actorRunId='98e7e40a-cd82-44c3-83b7-b3e2f76149d8')
phase='generation';done=threading.Event()
def report(text):
 payload=dict(BASE,operationId=str(uuid.uuid4()),kind='progress',text=text,proposedReportAt=(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(minutes=25)).isoformat())
 try:
  req=urllib.request.Request('http://100.87.176.23:5173/api/direct/requests/report',json.dumps(payload).encode(),{'Content-Type':'application/json'})
  response=urllib.request.urlopen(req,timeout=20).read().decode();(R/f'report-{payload["operationId"]}.json').write_text(response)
 except Exception as exc:print('REPORT DELIVERY ERROR',repr(exc),flush=True)
def heartbeat():
 while not done.wait(1500):
  count=sum(1 for _ in (R/'games').glob('[0-9]*-*.json.gz'));report(f'5時間追加試験継続: phase={phase}; 新規生成{count}/24768局。固定PLANの量・全3seed・dev選択・最終標本を維持。旧成果/main/Frozen/Webは変更なし。終了目安10:29UTC。')
def run(name,args,env=None):
 global phase
 phase=name;start=time.time();print('START',name,time.ctime(),flush=True)
 with (R/f'{name}.log').open('a') as h:subprocess.run([PY,*args],stdout=h,stderr=subprocess.STDOUT,check=True,env=env)
 print('DONE',name,round(time.time()-start),flush=True)
 times=json.loads((R/'phase-times.json').read_text()) if (R/'phase-times.json').exists() else [];times.append(dict(phase=name,seconds=time.time()-start,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()));(R/'phase-times.json').write_text(json.dumps(times,indent=2))
def main():
 threading.Thread(target=heartbeat,daemon=True).start()
 while not (R/'generation-telemetry.json').exists():time.sleep(10)
 assert json.loads((R/'generation-telemetry.json').read_text())['returncode']==0 and (R/'games/timing.json').exists()
 report('新24,576訓練局＋192dev局のGPU生成が完了。独立再生・新旧分離・特徴化を開始。追加学習と段階/独立対戦へ継続します。')
 if not (R/'data-audit.json').exists():run('features',['scripts/placement_extension_features.py'])
 if not (R/'evaluation-banks.json').exists():run('evaluation-banks',['scripts/placement_extension_eval.py','banks'])
 for count in (36864,49152):
  root=R/f'n{count}';env=dict(os.environ,PLACEMENT_STAGE=str(root))
  if not (root/'checkpoint-selection.json').exists():run(f'train-{count}',['scripts/monitor_generation.py','--telemetry',str(root/'training-telemetry.json'),PY,'scripts/placement_extension_train.py'],env)
  report(f'累積{count}局のA/C×全3seed実CUDA16epoch更新完了。新devだけで共通epoch固定。時間対戦中は生成/学習を重ねず、固定8prefix段階対戦へ。')
  if not (root/'export-check.json').exists():run(f'export-{count}',['scripts/check_extension_exports.py'],env)
  if not (root/'validation-summary.json').exists():run(f'validation-{count}',['scripts/placement_extension_eval.py','validation','--stage',str(count)])
  report(f'累積{count}局の固定段階比較完了。点推定だけで上限とはせず、最終候補49152局は事前固定を維持。')
 if not (R/'n49152/confirmation-summary.json').exists():run('confirmation',['scripts/placement_extension_eval.py','confirmation'])
 if not (R/'audit.json').exists():run('audit',['scripts/audit_placement_extension.py'])
 if not (R/'n49152/checks-post-training.json').exists():run('resume-check',['scripts/check_extension_resume.py'],dict(os.environ,PLACEMENT_STAGE=str(R/'n49152')))
 report('二段階の実学習・全1728局の新対戦・独立監査終了。最終推奨は補正CIで判定し、図表/CPU再現/保存/PR更新を仕上げます。')
 (R/'pipeline-complete.json').write_text(json.dumps(dict(complete=True,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())))
if __name__=='__main__':
 try:main()
 except BaseException as exc:report('追加試験外部pipeline停止: '+repr(exc)+'。完了扱いにせず未了段階から復旧。モデル/標本の救済変更はしません。');raise
 finally:done.set()
