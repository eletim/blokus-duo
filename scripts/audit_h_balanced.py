"""Audits after all timed matches. Old rule/history audits remain preserved."""
from h_balanced import *
from audit_h_distill import replay,audited_state
from blokus.ppo_value_leaf import value_only
from h_balanced_eval import aggregate,PAIRS

def audit():
 idx=np.load(INDEX,mmap_mode='r');assert hashlib.sha256(idx.tobytes()).hexdigest()=='cdd1f0f1ac7a0e722a670fb541ce904187949c6982cd03fda1fa7acad30f7b4c'
 forbidden=base.exclusions();sets={};n=0;label_replays=0;devdiagnostic={};countdoc=json.loads((R/'pre-training-counts.json').read_text());models=json.loads((R/'models.json').read_text())
 assert json.loads((R/'selection.json').read_text())['models_sha256']==sha(R/'models.json')
 for seed in SEEDS:assert models[str(seed)]['public']['sha256']==sha(f'artifacts/ppo-gen3/seed-{seed}.pt')
 for phase in ('pilot','train','dev'):
  hashes=set()
  for seed in SEEDS:
   j=json.loads((R/f'{phase}-{seed}.json').read_text());assert j['teacher_model_sha256']==sha(f'artifacts/ppo-gen3/seed-{seed}.pt');p=PPOAgent(seed,'cpu')
   for g in j['games']:
    assert replay(g['moves'])==g['scores'];n+=1;s=State();visited=[state_hash(s)]
    for k,m in enumerate(g['moves'],1):
     play(s,Move(*m) if m else None,validate=False);visited.append(state_hash(s))
     if k>=6:hashes.add(state_hash(s))
    assert visited==g['visited']
   for row in j['labels']:
    s=audited_state(row['moves']);assert row['state_hash']==state_hash(s) and row['state_hash'] not in forbidden and not contains(idx,row['state_hash']);assert row['stats']['depth']==6 and s.turn==row['turn']
    assert row['moves']==j['games'][row['game']]['moves'][:row['ply']]
    legal={tuple(raw(v.move)) for v in legal_moves(s)};assert len(legal)==row['legal_count']>=2 and tuple(row['target']) in legal and all(tuple(m) in legal for m in row['targets'])
    obs,qs=p._observation(s)
    with torch.no_grad():logits,_=p.model(*collate([obs],'cpu'));order=logits[0].argsort(descending=True).tolist()[:4]
    roots=[raw(qs[i].move) for i in order]
    if row['greedy'] not in roots:roots[-1]=row['greedy']
    scores=row['stats']['root_scores'];assert row['target']==roots[max(range(len(roots)),key=scores.__getitem__)]
    assert row['targets']==[m for m,v in zip(roots,scores) if abs(v-max(scores))<=1e-7]
   for row in j['labels'][::max(1,len(j['labels'])//3)]:
    t=DeepSearch(p,60000);assert raw(t.act(audited_state(row['moves']),random.Random(0)))==row['target'];np.testing.assert_allclose(t.stats['root_scores'],row['stats']['root_scores'],atol=1e-7,rtol=0);label_replays+=1
   if phase=='train':
    selected,_,counts=select(j['labels'],64);assert countdoc[str(seed)]['train']==counts
    for arm,rows in selected.items():assert rows==json.loads((R/f'{arm}-train-{seed}.json').read_text())['labels']
   if phase=='dev':
    _,rows,dc=select(j['labels'],16);assert rows==json.loads((R/f'balanced-dev-{seed}.json').read_text())['labels'];assert dc==countdoc[str(seed)]['dev']
    for arm in ('A','B'):
     q=PPOAgent(seed,'cpu',checkpoint=R/f'{arm}-{seed}.pt');assert all(torch.equal(v,q.model.state_dict()[k]) for k,v in p.model.state_dict().items() if not k.startswith('policy.'))
     assert models[str(seed)][arm]['sha256']==sha(R/f'{arm}-{seed}.pt');changes=[];drops=[];agreement=[];original=[];overlap=[]
     for row in rows:
      s=audited_state(row['moves']);assert value_only(p,s)==value_only(q,s)
      po,qs=p._observation(s)
      with torch.no_grad():a,av=p.model(*collate([po],'cpu'));b,bv=q.model(*collate([po],'cpu'))
      assert torch.equal(av,bv);oa=a[0].argsort(descending=True).tolist()[:4];ob=b[0].argsort(descending=True).tolist()[:4];rawqs=[raw(x.move) for x in qs]
      targets=[rawqs.index(m) for m in row['targets']];chosen=rawqs.index(row['target']);changes.append(oa[0]!=ob[0]);drops.append(chosen not in ob);agreement.append(ob[0] in targets);original.append(oa[0] in targets);overlap.append(len(set(oa)&set(ob))/len(oa))
     for device in ('cuda',):
      ca=PPOAgent(seed,device);cb=PPOAgent(seed,device,checkpoint=R/f'{arm}-{seed}.pt')
      for row in rows[::19]:assert value_only(ca,audited_state(row['moves']))==value_only(cb,audited_state(row['moves']))
     devdiagnostic[f'{arm}-{seed}']=dict(labels=len(rows),turn_counts={str(t):sum(r['turn']==t for r in rows) for t in (0,1)},agreement=float(np.mean(agreement)),original_agreement=float(np.mean(original)),greedy_change=float(np.mean(changes)),teacher_chosen_dropped_top4=sum(drops),top4_overlap=float(np.mean(overlap)),frozen_cpu_cuda_value_bit_equal=True,by_turn={str(turn):dict(agreement=float(np.mean([a for r,a in zip(rows,agreement) if r['turn']==turn])),original_agreement=float(np.mean([a for r,a in zip(rows,original) if r['turn']==turn])),labels=sum(r['turn']==turn for r in rows)) for turn in (0,1)})
  sets[phase]=hashes;print('coordinate labels',phase,n,flush=True)
 splits={f'{a}/{b}':len(sets[a]&sets[b]) for a,b in [('pilot','train'),('pilot','dev'),('train','dev')]};assert not any(splits.values());train=set.union(*sets.values());evalreport={};starts=set()
 for phase in ('validation','confirmation'):
  bank=json.loads((R/f'{phase}-bank.json').read_text())['prefixes'];assert all(p['state_hash'] not in train|forbidden|starts and not contains(idx,p['state_hash']) for p in bank);starts|={p['state_hash'] for p in bank};seen=set();over=0;games=0
  for method in list(PAIRS)+['policy-B-public','policy-A-public']:
   for seed in SEEDS:
    path=R/f'{phase}-{method}-{seed}.json'
    if not path.exists():continue
    row=json.loads(path.read_text())
    for g in row['games']:
     ms=bank[g['prefix']]['moves']+g['moves'];scores=replay(ms);assert scores==g['scores'] and g['difference']==scores[g['seat']]-scores[1-g['seat']];games+=1
     s=audited_state(bank[g['prefix']]['moves']);seen.add(state_hash(s))
     for m in g['moves']:
      play(s,Move(*m) if m else None,validate=False);h=state_hash(s);seen.add(h);over+=int(contains(idx,h))
  evalreport[phase]=dict(games=games,overlap_training=len(seen&train),original_training_hits=over,old_evaluation_start_hits=len(seen&forbidden));assert not seen&train and over==0 and not seen&forbidden
  sets[phase]=seen
  path=R/f'{phase}-summary.json'
  if path.exists():
   summary=json.loads(path.read_text());assert summary=={m:aggregate(phase,m) for m in summary}
 assert not sets['validation']&sets['confirmation']
 write(R/'audit.json',dict(pass_=True,coordinate_generation_games=n,label_tree_replays=label_replays,split_overlaps=splits,evaluation=evalreport,models_sha256=sha(R/'models.json'),pretrain_counts_sha256=sha(R/'pre-training-counts.json'),old_audit_reference='163791f:artifacts/h-policy-distill/audit.json'))
 write(R/'dev-diagnostics.json',devdiagnostic)
if __name__=='__main__':base.setup();audit()
