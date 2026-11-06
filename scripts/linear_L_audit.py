"""Independent-coordinate replay of the fixed L confirmation and its arithmetic."""
import hashlib,json
import numpy as np
from scripts import linear_L_confirm as ev,linear_search_audit as oracle,linear_value_eval as base
from blokus.linear_train2 import state_hash
R=ev.R

def audit():
 ev.check_models();bank=ev.read(R/'prefixes.json');pre=ev.read(R/'pre-audit.json');assert base.sha(R/'PLAN.md')==pre['plan_sha256'];assert bank['sha256']==hashlib.sha256(json.dumps(bank['prefixes'],sort_keys=True).encode()).hexdigest();starts={r['seed']:r['hash'] for r in bank['prefixes']};assert len(starts)==256
 old=np.load(R/'old-evaluation-states.npy',mmap_mode='r');idx=ev.indices()+[old];assert not set(starts.values())&set(ev.read(R/'old-evaluation.json')['excluded_starts']);rows=[];groups={};allstates=set()
 for a,record in zip(idx,pre['indices']):assert len(a)==record['states'] and hashlib.sha256(memoryview(a).cast('B')).hexdigest()==record['raw_sha256']
 for record in ev.read(R/'old-evaluation.json')['sources']:assert base.sha(record['path'])==record['sha256']
 for right in ('F','gen3'):
  groups[right]=[]
  for seed in ev.SEEDS:
   p=R/f'L-{right}-{seed}.json.gz';d=ev.read(p);assert d['prefix_sha256']==bank['sha256'];assert d['left_model_sha256']==base.sha(ev.source('L',seed)) and d['right_model_sha256']==base.sha(ev.source(right,seed));visited=set();seats=set()
   for g in d['games']:
    s,_=oracle.start(g['prefix_seed']);assert state_hash(s)==starts[g['prefix_seed']];assert (g['prefix_seed'],g['seat']) not in seats;seats.add((g['prefix_seed'],g['seat']));oracle.replay(g,visited)
   assert len(seats)==512;keys=np.array(sorted(bytes.fromhex(h) for h in visited),dtype='S32')
   for a in idx:
    ii=np.searchsorted(a,keys);valid=ii<len(a);assert not np.any(a[ii[valid]]==keys[valid])
   for st in d['left_search']+d['right_search']:
    if st.get('forced_pass'):continue
    if st['fallback']:assert st['values']==[]
    if st['complete']:assert len(st['values'])==st['root_candidates']==st['completed_roots']
    comp=[a for a in st['attempts'] if a['complete']];assert st['depth']==(comp[-1]['depth'] if comp else 0)
    assert all(a['roots']==a['completed_roots'] for a in comp)
    if st['roots']:assert st['baseline'] in st['roots'] and len(st['roots'])==len(set(map(tuple,st['roots'])))
   groups[right].append(d);allstates.update(visited);rows.append(dict(path=str(p),sha256=base.sha(p),games=len(d['games']),states=len(visited),training_old_eval_overlap=0))
 expected=ev.read(R/'results.json');decision=ev.read(R/'decision.json')
 for right,ds in groups.items():
  assert ev.summarize(ds)==expected['L/'+right]
  games=sum([d['games'] for d in ds],[]);wins=sum(g['difference']>0 for g in games);draws=sum(g['difference']==0 for g in games);assert abs((wins+draws/2)/len(games)-expected['L/'+right]['point'])<1e-14
  # Independent matrix arithmetic for the two primary cluster intervals.
  matrix=np.array([[g['difference'] for d in ds for g in d['games'] if g['prefix_seed']==p['seed']] for p in bank['prefixes']]);assert matrix.shape==(256,6)
  points=((matrix>0)+.5*(matrix==0)).mean(axis=1);diffs=matrix.mean(axis=1);draw=np.random.default_rng(2490000000).integers(256,size=(20000,256));out=expected['L/'+right]
  assert np.allclose(np.quantile(points[draw].mean(axis=1),[.0125,.9875]),out['point_ci_bonferroni2'],atol=1e-14,rtol=0)
  assert np.allclose(np.quantile(diffs[draw].mean(axis=1),[.025,.975]),out['difference_ci95'],atol=1e-14,rtol=0)
 assert decision['old_nonNN_superiority']==(expected['L/F']['point_ci_bonferroni2'][0]>.5) and decision['gen3_superiority']==(expected['L/gen3']['point_ci_bonferroni2'][0]>.5)
 ev.write('audit.json',dict(pass_=True,rows=rows,games=sum(x['games'] for x in rows),coordinate_oracle=True,scores_rewards=True,arithmetic_CI=True,model_sha=True,training_old_eval_overlap=0,visited_states=len(allstates),complete_stage_fallback=True));print('AUDIT PASS',flush=True)
if __name__=='__main__':audit()
