"""Run after timed experiments, never concurrently with them."""
import json
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from blokus.value_experiment import write
from blokus.research import sha


def main():
    root=Path('artifacts/visited-diversity');assert json.loads((root/'summary.json').read_text())['completed']
    records=[]
    for args in [['-m','pytest','-q'],['scripts/check_visited_resume.py'],['scripts/audit_visited.py'],['scripts/diagnose_visited.py']]:
        start=time.perf_counter();r=subprocess.run([sys.executable,*args],text=True,capture_output=True,timeout=1200)
        records.append(dict(command=[sys.executable,*args],returncode=r.returncode,seconds=time.perf_counter()-start,stdout=r.stdout,stderr=r.stderr))
        print(args,r.returncode,r.stdout[-1000:],flush=True)
        if r.returncode:
            write(root/'verification-partial.json',records);raise RuntimeError(args)
    spec=json.loads((root/'play-specs.json').read_text())['visited-601']
    cmd=[sys.executable,'-m','blokus.value_experiment','play','--checkpoint',spec['checkpoint'],'--alpha','.5','--gate','24','--budget-ms','6']
    start=time.perf_counter();r=subprocess.run(cmd,input='0\n'*100,text=True,capture_output=True,timeout=120)
    assert r.returncode==0 and 'scores ' in r.stdout
    records.append(dict(command=cmd,returncode=r.returncode,seconds=time.perf_counter()-start,stdout=r.stdout[r.stdout.rfind('scores '):],stderr=r.stderr))
    write(root/'verification.json',dict(status='PASS',checks=records,torch=torch.__version__,gpu=torch.cuda.get_device_name(),source_hashes={str(p):sha(p) for folder in ['blokus','scripts','tests'] for p in sorted(Path(folder).glob('*.py'))}))
    print('VERIFIED',flush=True)
if __name__=='__main__':main()
