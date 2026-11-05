"""Materialize exact placements, retaining only known labels for training/evaluation."""
import gzip,json
import numpy as np
from .mask_data import ROOT
from .engine import Move
from .research import sha
from .value_experiment import write

def main():
 raw=np.load(ROOT/'data-2400.npz');d={k:raw[k] for k in raw.files};records=json.load(gzip.open(ROOT/'teacher-labels.json.gz','rt'));assert len(records)==len(d['x'])
 cells=np.zeros((len(records),128,196),np.uint8);collision=None
 for j,r in enumerate(records):
  assert r['game']==d['game'][j]
  seen={}
  for i,m in enumerate(r['moves']):
   for y,x in Move(*m).cells:cells[j,i,y*14+x]=1
   key=d['x'][j,i].tobytes()
   if key in seen and not r['bank'] and collision is None:
    k=seen[key]
    if not np.array_equal(cells[j,k],cells[j,i]):collision=dict(state=r['state'],state_hash=r['state_hash'],game=r['game'],moves=[r['moves'][k],m],summary=d['x'][j,i].tolist())
   seen[key]=i
 write(ROOT/'collision.json',collision)
 # Every labeled candidate once. Padding remains unknown, never a zero target.
 ids=np.zeros((len(records),6),int);valid=np.zeros_like(ids,bool)
 for j in range(len(records)):
  ix=np.flatnonzero(d['mask'][j]);ids[j,:len(ix)]=ix;valid[j,:len(ix)]=True
 rows=np.arange(len(records))[:,None]
 for k in ['x','target','initial','random']:d[k]=d[k][rows,ids]
 d['x']=np.concatenate((d['x'],cells[rows,ids].astype(np.float32)),2);d['mask']=valid
 np.savez_compressed(ROOT/'compact.npz',**d)
 write(ROOT/'prepared.json',dict(data_sha256=sha(ROOT/'compact.npz'),roots=len(records),training_games=len(set(d['game'][~d['bank']].tolist())),evaluation_games=len(set(d['game'][d['bank']].tolist())),collision_game=collision['game']))
if __name__=='__main__':main()
