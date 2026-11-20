import json,hashlib,subprocess,sys
from pathlib import Path
R=Path('artifacts/gpu-async512');m=json.loads((R/'steady-bank-manifest.json').read_text())
for p,h in [(R/'steady-bank.json',m['bank_sha']),(R/'steady-excluded.json.gz',m['excluded_sha'])]:assert hashlib.sha256(p.read_bytes()).hexdigest()==h
subprocess.run([sys.executable,'scripts/audit_async.py','--bank',str(R/'steady-bank.json'),'--output',str(R/'steady-audit.json'),'--exclusions',str(R/'steady-excluded.json.gz'),str(R/'steady-4096')],check=True)
subprocess.run([sys.executable,'scripts/summarize_async_steady.py'],check=True)
