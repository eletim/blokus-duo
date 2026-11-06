import gzip,json,os,subprocess,sys,time
from pathlib import Path
R=Path('artifacts/linear-tuple');python=os.environ.get('BLOKUS_CPU_PYTHON',sys.executable);rows=[]
for backend,seat in [('native',None),('python',0),('python',1)]:
 cmd=[python,'-m','blokus.tuple_play','--checkpoint',str(R/'T-3101.json'),'--backend',backend]+([] if seat is None else ['--human-seat',str(seat)]);start=time.perf_counter();p=subprocess.run(cmd,input='invalid\n'+'0\n'*100,text=True,capture_output=True,timeout=120,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'});assert p.returncode==0,p.stderr;d=json.loads(p.stdout.splitlines()[-1]);assert len(d['scores'])==2 and len(d['moves'])>0
 if seat is not None:assert '表示された合法手番号' in p.stdout
 (R/f'cli-{backend}-{seat}.log.gz').write_bytes(gzip.compress(p.stdout.encode(),mtime=0));rows.append(dict(command=cmd,wall=time.perf_counter()-start,scores=d['scores'],moves=len(d['moves']),human_input_validated=seat is not None))
(R/'cli-audit.json').write_text(json.dumps(rows,indent=2)+'\n')
