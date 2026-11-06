import json,os,subprocess,time,sys
from pathlib import Path
p=os.environ.get('BLOKUS_CPU_PYTHON',sys.executable);base=[p,'-m','blokus.linear_play2','--checkpoint','artifacts/linear-hybrid/exports/seed-3101.json','--completed-search','--search-ms','100','--root-width','24','--reply-width','8','--depth','2'];rows=[]
for backend,seat in [('native',None),('python',0),('python',1)]:
 cmd=base+['--backend',backend]+([] if seat is None else ['--human-seat',str(seat)]);t=time.perf_counter();r=subprocess.run(cmd,input='bad\n'+'0\n'*100,text=True,capture_output=True,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'},timeout=90);assert r.returncode==0,r.stderr;line=r.stdout.splitlines()[-1];d=json.loads(line);assert len(d['scores'])==2 and len(d['moves'])>=2;rows.append(dict(command=cmd,seconds=time.perf_counter()-t,returncode=r.returncode,scores=d['scores'],moves=len(d['moves']),invalid_input_reprompt=('表示された合法手番号' in r.stdout) if seat is not None else None));Path(f'artifacts/linear-search/cli-{backend}-{seat}.log').write_text(r.stdout)
Path('artifacts/linear-search/cli-audit.json').write_text(json.dumps(dict(cpu_only=True,rows=rows),indent=2)+'\n');print(rows)
