"""One bounded fit on saved records; disjoint prefix/paired-game components."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from blokus.engine import State,Move,play
from blokus.margin_value import features,FORMAT
R=Path('artifacts/margin-pilot');model_path=R/'model.json'
if model_path.exists():raise SystemExit('Model exists: refusing duplicate training; inspect saved result.')
rows=json.loads((R/'records.json').read_text());start=time.monotonic()
parent={r['cluster']:r['cluster'] for r in rows}
def find(a):
 while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
 return a
def union(a,b):
 a,b=find(a),find(b)
 if a!=b:parent[max(a,b)]=min(a,b)
data=[];owners={}
for r in rows:
 s=State();samples=[]
 for ply,m in enumerate(r['moves']):
  if ply>=len(r['prefix']) and not s.done:
   key=(tuple(s.boards),tuple(s.remaining),tuple(s.last),s.turn,s.passes)
   if key in owners:union(r['cluster'],owners[key])
   else:owners[key]=r['cluster']
   samples.append((features(s),(r['scores'][s.turn]-r['scores'][1-s.turn]),key))
  play(s,Move(*m) if m is not None else None,validate=False)
 data.append(samples)
components=sorted({find(k) for k in parent},key=lambda x:hashlib.sha256(('margin-split-20261127:'+x).encode()).hexdigest())
n=len(components);split={g:('train' if i<int(n*.6) else 'validation' if i<int(n*.8) else 'test') for i,g in enumerate(components)}
sets={k:set() for k in ('train','validation','test')};arrays={k:[] for k in sets};labels={k:[] for k in sets};weights={k:[] for k in sets};groups={k:[] for k in sets}
for r,samples in zip(rows,data):
 group=find(r['cluster']);name=split[group];r['split']=name;r['component']=group
 for x,y,key in samples:
  sets[name].add(key);arrays[name].append(x);labels[name].append(y);weights[name].append(1/len(samples));groups[name].append(group)
assert not (sets['train']&sets['validation'] or sets['train']&sets['test'] or sets['validation']&sets['test'])
X={k:np.array(v) for k,v in arrays.items()};Y={k:np.array(v) for k,v in labels.items()};W={k:np.array(v) for k,v in weights.items()}
metrics=lambda y,p,w:dict(mae=float(np.average(np.abs(p-y),weights=w)),bias=float(np.average(p-y,weights=w)),rmse=float(np.sqrt(np.average((p-y)**2,weights=w))))
xx=X['train'].T@(W['train'][:,None]*X['train']);xy=X['train'].T@(W['train']*Y['train']);candidates=[]
for alpha in (.01,.1,1.,10.):
 coef=np.linalg.solve(xx+alpha*np.eye(xx.shape[0]),xy);pred=np.clip(X['validation']@coef,-109,109)
 candidates.append((metrics(Y['validation'],pred,W['validation'])['mae'],alpha,coef))
_,alpha,coef=min(candidates,key=lambda r:(r[0],r[1]))
model=dict(format=FORMAT,weights=coef.tolist(),alpha=alpha,unit='raw final score points, mover perspective',
 training_source_sha256=hashlib.sha256((R/'records.json').read_bytes()).hexdigest(),
 selected_using='validation MAE only; equal total weight per game',candidates=[dict(alpha=a,validation_mae=m) for m,a,c in candidates])
# Freeze selection before inspecting held-out test predictions.
model_path.write_text(json.dumps(model,indent=2))
result={}
for name in X:
 pred=np.clip(X[name]@coef,-109,109);m=metrics(Y[name],pred,W[name]);m['positions']=len(pred);m['games']=sum(r['split']==name for r in rows);m['components']=len(set(groups[name]))
 # Compare observed current margin, not a second learned predictor.
 naive=X[name][:,53]+X[name][:,110]+X[name][:,167]
 m['current_margin_baseline']=metrics(Y[name],naive*89,W[name])
 group_stats=[]
 for g in sorted(set(groups[name])):
  ix=np.array(groups[name])==g;w=W[name][ix];err=pred[ix]-Y[name][ix]
  group_stats.append([sum(np.abs(err)*w),sum(err*w),sum(w)])
 a=np.array(group_stats);rng=np.random.default_rng(20261127);b=a[rng.integers(len(a),size=(2000,len(a)))].sum(1)
 m['mae_cluster_ci95']=np.quantile(b[:,0]/b[:,2],[.025,.975]).tolist();m['bias_cluster_ci95']=np.quantile(b[:,1]/b[:,2],[.025,.975]).tolist();result[name]=m
(R/'split-records.json').write_text(json.dumps(rows))
(R/'training-result.json').write_text(json.dumps(dict(metrics=result,alpha=alpha,seconds=time.monotonic()-start,
 split_components={k:sum(v==k for v in split.values()) for k in sets},state_intersections=0,features=xx.shape[0],model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest()),indent=2))
# Benchmark tasks fixed by group identity, not by test outcomes.
testgroups=sorted({r['component'] for r in rows if r['split']=='test'})[:12];tasks=[]
for i,g in enumerate(testgroups):
 row=next(r for r in rows if r['component']==g)
 for seat in (0,1):tasks.append(dict(id=f'paired-{i:02}-{seat}',kind='paired',seed=1751401+i%3,seat=seat,prefix=row['prefix'],cluster=g,rng_seed=2026112700+i,komi=0))
for i,seed in enumerate((1751401,1751402,1751403)):
 for seat in (0,1):tasks.append(dict(id=f'self-{seed}-{seat}',kind='self',seed=seed,seat=seat,prefix=[],cluster=f'seed-{seed}',rng_seed=2026112800+i,komi=0))
(R/'tasks.json').write_text(json.dumps(tasks,indent=2))
print((R/'training-result.json').read_text())
