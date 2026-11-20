"""Record external utilization while one explicitly requested child runs."""
import argparse,subprocess,time,json
import os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--telemetry',required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args();start=time.perf_counter();rows=[]
child=subprocess.Popen(a.command)
try:
 while child.poll() is None:
  text=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True).strip()
  stats=[];pids=[child.pid]
  for pid in pids:
   try:
    fields=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()
    stats.append(dict(pid=pid,cpu_seconds=(int(fields[11])+int(fields[12]))/os.sysconf('SC_CLK_TCK'),rss=int(fields[21])*os.sysconf('SC_PAGE_SIZE')))
    pids.extend(int(x) for x in Path(f'/proc/{pid}/task/{pid}/children').read_text().split())
   except (OSError,IndexError,ValueError):pass
  rows.append(dict(seconds=time.perf_counter()-start,gpu=text,processes=stats));time.sleep(1)
finally:
 Path(a.telemetry).write_text(json.dumps(dict(outer_seconds=time.perf_counter()-start,returncode=child.wait(),samples=rows),indent=2))
raise SystemExit(child.returncode)
