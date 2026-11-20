"""Sequential finite experiment, started after GPU generation; never deploys."""
import os,sys,time,json,subprocess,urllib.request,uuid,datetime
from pathlib import Path
R=Path('artifacts/value-placement-24576');PY=sys.executable
BASE=dict(humanRequestId='51a48411-ad31-4343-a063-fb0b56ee9c80',workId='b3c46e23-03a5-413f-876d-00c836bac6ee',actorRunId='71920aed-6131-435f-814e-6683f8ec9f5b')
def report(text):
 payload=dict(BASE,operationId=str(uuid.uuid4()),kind='progress',text=text,proposedReportAt=(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(minutes=30)).isoformat())
 try:
  req=urllib.request.Request('http://100.87.176.23:5173/api/direct/requests/report',json.dumps(payload).encode(),{'Content-Type':'application/json'});response=urllib.request.urlopen(req,timeout=20).read().decode();(R/f'report-{payload["operationId"]}.json').write_text(response)
 except Exception as exc:print('REPORT DELIVERY ERROR',repr(exc),flush=True)
def run(name,args):
 start=time.time();print('START',name,time.ctime(),flush=True)
 with (R/f'{name}.log').open('a') as h:subprocess.run([PY,*args],stdout=h,stderr=subprocess.STDOUT,check=True)
 print('DONE',name,round(time.time()-start),flush=True)
def main():
 while not (R/'generation-telemetry.json').exists():time.sleep(10)
 telemetry=json.loads((R/'generation-telemetry.json').read_text());assert telemetry['returncode']==0 and (R/'games/timing.json').exists()
 report('新18,432訓練局＋192開発局のGPU生成が終了。独立合法性再生・特徴作成へ進行。速度再実験や追加条件探索はしていません。')
 if not (R/'data-manifest.json').exists():run('features',['scripts/placement_large_features.py'])
 if not (R/'checks-pre-training.json').exists():run('pre-training-checks',['scripts/check_placement_large.py'])
 report('全新棋譜の独立再生・分離・終局WDLラベル検査、およびCPU/CUDA別processのAdam/RNG再開検査を通過。A/C×3seed、最大16epochの実CUDA更新を開始します。')
 if not (R/'checkpoint-selection.json').exists():run('training',['scripts/monitor_generation.py','--telemetry',str(R/'training-telemetry.json'),PY,'scripts/placement_large_train.py'])
 if not (R/'checks-post-training.json').exists():run('post-training-checks',['scripts/check_placement_large.py'])
 report('A/C全3seedの16epoch実CUDA学習と保存再開検査が終了。新devだけで共通epochを固定し、CPU500msの新validationへ進みます。')
 if not (R/'evaluation-manifest.json').exists():run('evaluation-bank',['scripts/placement_large_eval.py','banks'])
 if not (R/'selection.json').exists():run('validation',['scripts/placement_large_eval.py','validation'])
 sel=json.loads((R/'selection.json').read_text());report('新validation全288局終了。事前screen='+str(sel['passed'])+'。通過時のみ固定32prefixの独立確認を実施し、未通過なら救済条件を追加せず監査へ進みます。')
 if sel['passed'] and not (R/'decision.json').exists():run('confirmation',['scripts/placement_large_eval.py','confirmation'])
 if not (R/'audit.json').exists():run('audit',['scripts/audit_placement_large.py'])
 report('実学習・必要な新対戦と独立棋譜/得点/集計/分離監査が終了。CPUクリーンcheckout、図表、REPORT、PR更新を仕上げます。')
 (R/'pipeline-complete.json').write_text(json.dumps(dict(complete=True,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())))
if __name__=='__main__':
 try:main()
 except BaseException as exc:
  report('有限実験pipelineが停止: '+repr(exc)+'。完了扱いにせず原因確認・修正して未了段階から再開します。');raise
