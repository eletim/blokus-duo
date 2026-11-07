"""Post-timing audit: independent coordinate replay, label replay, split and value checks."""
from h_distill import *
import importlib.util
spec=importlib.util.spec_from_file_location('oracle','artifacts/secretary-review/blokus_reference.py');oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)
from blokus.engine import PIECES
from blokus.ppo_value_leaf import value_only

def replay(ms):
 own=[set(),set()];remaining=[set(range(21)),set(range(21))];last=[None,None];turn=0;passes=0
 for rawmove in ms:
  assert passes<2
  if rawmove is None:
   assert not any(oracle.legal_cells(PIECES[i],own[turn],own[1-turn],oracle.STARTS[turn]) for i in remaining[turn]);passes+=1
  else:
   m=Move(*rawmove);assert m.piece in remaining[turn] and oracle.valid(m.cells,own[turn],own[1-turn],oracle.STARTS[turn]);own[turn].update(m.cells);remaining[turn].remove(m.piece);last[turn]=m.piece;passes=0
  turn=1-turn
 assert passes==2
 return [-sum(len(PIECES[i]) for i in rem) if rem else (20 if last[p]==0 else 15) for p,rem in enumerate(remaining)]

def audited_state(ms):
 s=State()
 for m in ms:play(s,Move(*m) if m else None,validate=False)
 return s

def audit():
 idx=np.load(INDEX,mmap_mode='r');assert sha(INDEX) # npy file includes header; raw array hash below
 assert hashlib.sha256(idx.tobytes()).hexdigest()=='cdd1f0f1ac7a0e722a670fb541ce904187949c6982cd03fda1fa7acad30f7b4c'
 forbidden=exclusions();sets={};n=0;label_replays=0;frozen_checks=[]
 for phase in ('pilot','train','dev'):
  hashes=set()
  for seed in SEEDS:
   j=json.loads((R/f'{phase}-{seed}.json').read_text());assert j['teacher_model_sha256']==sha(f'artifacts/ppo-gen3/seed-{seed}.pt');p=PPOAgent(seed,'cpu');q=PPOAgent(seed,'cpu',checkpoint=R/f'student-{seed}.pt')
   for g in j['games']:
    assert replay(g['moves'])==g['scores'];n+=1
    s=State()
    for k,m in enumerate(g['moves'],1):
     play(s,Move(*m) if m else None,validate=False)
     if k>=6:hashes.add(state_hash(s))
   for row in j['labels']:
    s=audited_state(row['moves']);assert row['state_hash']==state_hash(s) and row['state_hash'] not in forbidden and not contains(idx,row['state_hash']);assert row['stats']['depth']==6
    assert row['moves']==j['games'][row['game']]['moves'][:row['ply']]
    legal={tuple(raw(v.move)) for v in legal_moves(s)};assert tuple(row['target']) in legal and all(tuple(m) in legal for m in row['targets'])
    assert value_only(p,s)==value_only(q,s)
    obs,qs=p._observation(s)
    with torch.no_grad():logits,_=p.model(*collate([obs],'cpu'));order=logits[0].argsort(descending=True).tolist()[:4]
    roots=[raw(qs[i].move) for i in order]
    if row['greedy'] not in roots:roots[-1]=row['greedy']
    scores=row['stats']['root_scores'];assert row['target']==roots[max(range(len(roots)),key=scores.__getitem__)]
    assert row['targets']==[m for m,v in zip(roots,scores) if abs(v-max(scores))<=1e-7]
   # Replay a stratified small label subset, same finite completed tree.
   for row in j['labels'][::max(1,len(j['labels'])//3)]:
    t=DeepSearch(p,60000);assert raw(t.act(audited_state(row['moves']),random.Random(0)))==row['target'];np.testing.assert_allclose(t.stats['root_scores'],row['stats']['root_scores'],atol=1e-7,rtol=0);label_replays+=1
   if phase=='dev':
    for device in ('cpu','cuda'):
     a=PPOAgent(seed,device);b=PPOAgent(seed,device,checkpoint=R/f'student-{seed}.pt')
     for row in j['labels'][::max(1,len(j['labels'])//5)]:assert value_only(a,audited_state(row['moves']))==value_only(b,audited_state(row['moves']))
    original=p.model.state_dict();student=q.model.state_dict();assert all(torch.equal(v,student[k]) for k,v in original.items() if not k.startswith('policy.'));frozen_checks.append(seed)
  sets[phase]=hashes;print('coordinate+label audit',phase,n,flush=True)
 splits={f'{a}/{b}':len(sets[a]&sets[b]) for a,b in [('pilot','train'),('pilot','dev'),('train','dev')]};assert not any(splits.values())
 evalreport={};train=sets['pilot']|sets['train']|sets['dev'];starts=set()
 from h_distill_eval import aggregate
 for phase in ('validation','confirmation'):
  bank=json.loads((R/f'{phase}-bank.json').read_text())['prefixes'];assert all(p['state_hash'] not in train|forbidden|starts and not contains(idx,p['state_hash']) for p in bank);starts|={p['state_hash'] for p in bank};seen=set();over=0;games=0
  for method in ('policy','H'):
   for seed in SEEDS:
    path=R/f'{phase}-{method}-{seed}.json'
    if not path.exists():continue
    row=json.loads(path.read_text())
    for g in row['games']:
     ms=bank[g['prefix']]['moves']+g['moves'];scores=replay(ms);assert scores==g['scores'];assert g['difference']==scores[g['seat']]-scores[1-g['seat']];games+=1
     s=audited_state(bank[g['prefix']]['moves']);seen.add(state_hash(s))
     for m in g['moves']:play(s,Move(*m) if m else None,validate=False);h=state_hash(s);seen.add(h);over+=int(contains(idx,h))
  evalreport[phase]=dict(games=games,overlap_training=len(seen&train),original_training_hits=over);assert not seen&train and over==0
  sets[phase]=seen;print('coordinate+evaluation audit',phase,games,flush=True)
 assert not sets['validation']&sets['confirmation']
 for phase in ('validation','confirmation'):
  path=R/f'{phase}-summary.json'
  if path.exists():assert json.loads(path.read_text())=={m:aggregate(phase,m) for m in ('H','policy')}
 write(R/'audit.json',dict(pass_=True,coordinate_training_games=n,label_tree_replays=label_replays,split_overlaps=splits,evaluation=evalreport,frozen_value_cpu_cuda_bit_equal=frozen_checks,source_model_hashes={str(s):sha(f'artifacts/ppo-gen3/seed-{s}.pt') for s in SEEDS}))
if __name__=='__main__':setup();audit()
