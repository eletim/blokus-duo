"""Arithmetic of the new experiment, independent of actual match outcomes."""
import gzip,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import placement_large_eval as q

def test_prefix_clusters_combine_seats_and_all_seeds(tmp_path,monkeypatch):
 monkeypatch.setattr(q,'R',tmp_path);monkeypatch.setattr(q.e,'R',tmp_path)
 for si,seed in enumerate(q.e.SEEDS):
  games=[]
  for prefix in range(4):
   for seat in (0,1):
    diff=(1-si)*4
    stats=[dict(method=name,forced_pass=False,elapsed_ms=100.,depth=6,fallback=False) for name in 'AQ']
    games.append(dict(prefix=prefix,seat=seat,difference=diff,point=float(diff>0)+.5*(diff==0),stats=stats))
  with gzip.open(tmp_path/f'validation-AQ-{seed}.json.gz','wt') as h:json.dump(dict(games=games),h)
 out=q.aggregate('validation','AQ')
 assert out['games']==24 and (out['wins'],out['draws'],out['losses'])==(8,8,8)
 assert out['point']==.5 and out['point_ci95']==[.5,.5] and out['point_adjusted_ci']==[.5,.5]
 assert out['difference']==0 and out['difference_adjusted_ci']==[0,0]
 assert out['by_seed']=={str(s):v for s,v in zip(q.e.SEEDS,[1.,.5,0.])}
 assert all(v['point']==.5 for v in out['by_seat'].values())
 assert out['stats']['A']['mean_ms']==100 and out['stats']['A']['depth6']==1
