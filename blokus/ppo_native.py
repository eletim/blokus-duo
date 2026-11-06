"""Optional ordered native PPO observation backend with pure Python fallback."""
import argparse,ctypes,hashlib,json,shutil,subprocess,sys,threading,time
from functools import lru_cache
from pathlib import Path
import numpy as np
from .engine import PLACEMENTS,SIZES,frontier,legal_moves
from .placement_geometry import geometry
from .ppo_policy import observe as python_observe
from .value_search import encode
ROOT=Path(__file__).resolve().parents[1];SOURCE=Path(__file__).with_suffix('.c');BUILD=ROOT/'.native/ppo'
LOCAL=threading.local()
def digest():return hashlib.sha256(SOURCE.read_bytes()).hexdigest()
def build():
 BUILD.mkdir(parents=True,exist_ok=True);compiler=next((shutil.which(c) for c in ('cc','gcc','clang') if shutil.which(c)),None)
 cmd=[compiler] if compiler else [sys.executable,'-m','ziglang','cc']
 start=time.perf_counter();subprocess.run(cmd+['-O3','-fPIC','-shared','-std=c11','-march=native',str(SOURCE),'-o',str(BUILD/'backend.so')],check=True)
 data=dict(source_sha256=digest(),compiler=cmd,seconds=time.perf_counter()-start,flags='-O3 -fPIC -shared -std=c11 -march=native; no fast-math',host_specific=True);(BUILD/'build.json').write_text(json.dumps(data,indent=2));print(json.dumps(data));return data
@lru_cache(maxsize=1)
def library():
 try:
  meta=json.loads((BUILD/'build.json').read_text())
  if meta['source_sha256']!=digest():return None
  lib=ctypes.CDLL(str(BUILD/'backend.so'));lib.ppo_scan.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int];lib.ppo_scan.restype=ctypes.c_int;return lib
 except (OSError,ValueError,KeyError):return None

def words(x):return [(x>>(64*i))&((1<<64)-1) for i in range(4)]
@lru_cache(maxsize=1)
def tables():
 n=len(PLACEMENTS);bits=np.array([words(q.bits)+words(q.edges)+words(q.corners) for q in PLACEMENTS],np.uint64);pieces=np.array([q.move.piece for q in PLACEMENTS],np.int32);fixed=np.zeros((2,n,39),np.float32);cells=np.empty((n,5),np.int64)
 for i,q in enumerate(PLACEMENTS):
  center,reach,extra,ix=geometry(q.move);cells[i]=ix
  for p in [0,1]:fixed[p,i,0]=SIZES[q.move.piece]/5;fixed[p,i,3]=center;fixed[p,i,4]=reach[p];fixed[p,i,8]=float(SIZES[q.move.piece]==1);fixed[p,i,12:]=extra
 for a in (bits,pieces,fixed,cells):a.flags.writeable=False
 return bits,pieces,fixed,cells

def scan(s,pack=True):
 lib=library()
 if lib is None:raise RuntimeError('native backend unavailable: run python -m blokus.ppo_native build (cc or optional ziglang required)')
 bits,pieces,fixed,cells=tables();n=len(PLACEMENTS)
 if not hasattr(LOCAL,'ids'):LOCAL.ids=np.empty(n,np.int32);LOCAL.x=np.empty((n,39),np.float32)
 if s.done:return np.empty(0,np.int32),np.empty((0,39),np.float32)
 p=s.turn;f,e=frontier(s,p);of,_=frontier(s,1-p);ctx=np.array(words(s.boards[p])+words(s.boards[1-p])+words(f)+words(of)+words(e),np.uint64)
 count=lib.ppo_scan(ctx.ctypes.data,s.remaining[p],n,bits.ctypes.data,pieces.ctypes.data,fixed[p].ctypes.data,LOCAL.ids.ctypes.data,LOCAL.x.ctypes.data,int(pack))
 return LOCAL.ids[:count].copy(),LOCAL.x[:count].copy() if pack else None

def observe(s,channels=4,backend='auto',timing=None):
 if backend not in ('python','native','auto'):raise ValueError('unknown backend')
 if channels not in (4,6):raise ValueError('channels must be 4 or 6')
 if s.done:raise ValueError('terminal has no actions')
 use=backend=='native' or (backend=='auto' and library() is not None)
 start=time.perf_counter()
 if use:
  ids,x=scan(s);qs=[PLACEMENTS[i] for i in ids];mid=time.perf_counter();b,st=encode(s);b=b[:channels].copy()
  if len(ids):cells=tables()[3][ids].copy()
  else:x=np.zeros((1,39),np.float32);cells=np.full((1,5),196,np.int64)
  obs=(b,st,x,cells)
 else:
  qs=legal_moves(s);mid=time.perf_counter();obs,_=python_observe(s,qs,channels)
 if timing is not None:
  timing['legal_and_native_features' if use else 'legal']+=mid-start;timing['pack']+=time.perf_counter()-mid;timing['candidates']+=len(qs)
 return obs,qs

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['build']);p.parse_args();build()
