"""Serial driver: no audit, profiling or learning overlaps timed generation."""
import os,subprocess,sys
from pathlib import Path
r=Path('artifacts/gpu-generation')
for mode,repeat in [('cpu',1),('gpu',1),('gpu',2),('cpu',2)]:
    result=r/f'{mode}-{repeat}.json'
    if result.exists():raise RuntimeError(f'Preserve existing result: {result}')
    print('starting',mode,repeat,flush=True)
    monitor=None;stream=None
    try:
        if mode=='gpu':
            stream=(r/f'{mode}-{repeat}-telemetry.csv').open('w')
            monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw','--format=csv','-l','1'],stdout=stream)
        with (r/f'{mode}-{repeat}.log').open('w') as log:
            subprocess.run([sys.executable,'scripts/gpu_generation.py',mode,'--repeat',str(repeat)],stdout=log,stderr=subprocess.STDOUT,check=True)
    finally:
        if monitor is not None:monitor.terminate();monitor.wait()
        if stream is not None:stream.close()
    print('completed',mode,repeat,flush=True)
