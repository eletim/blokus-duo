import copy,random
import torch
from blokus.gen4 import load
from blokus.engine import State,play,legal_moves
from blokus.candidate_teacher import RootWideSearch
from blokus.ppo_search_deeper import DeepSearch,key
from blokus.ppo_policy import collate
from candidate_train import objective

def test_fixed_tree_and_root_label_order():
 torch.set_num_threads(1);p,v=load(1751401,'native');s=State();rng=random.Random(9939)
 chosen=copy.deepcopy(s)
 for _ in range(28):
  if s.done:break
  q=legal_moves(s)
  if len(q)>=2:chosen=copy.deepcopy(s)
  play(s,rng.choice(q).move if q else None,validate=True)
 s=chosen;before=key(s);a=RootWideSearch(p,60000,max_depth=4,root_width=12,value_evaluator=v);b=RootWideSearch(p,60000,max_depth=4,root_width=12,value_evaluator=v,prune=False)
 assert a.act(s,rng)==b.act(s,rng);assert a.stats['root_scores']==b.stats['root_scores'];assert key(s)==before
 roots=a.stats['root_moves'];assert len({tuple(x) for x in roots})==len(roots)
 forced=State(remaining=[0,0]);assert a.act(forced,rng) is None;assert a.stats['forced_pass']
 terminal=State(remaining=[0,0],passes=2,last=[0,1]);assert v(terminal)==1

def test_head_update_and_exact_adam_resume(tmp_path):
 torch.set_num_threads(1);p,v=load(1751401,'native');s=State();obs,qs=p._observation(s);before=copy.deepcopy(p.model.state_dict());vv=v(s)
 with torch.no_grad():l,oldv=p.model(*collate([obs],'cpu'))
 target=torch.zeros(len(qs));target[-1]=1;row=dict(obs=obs,target=target,original=torch.log_softmax(l[0],0))
 for name,x in p.model.named_parameters():x.requires_grad_(name.startswith('policy.'))
 opt=torch.optim.Adam([x for x in p.model.parameters() if x.requires_grad],lr=1e-4)
 def step(m,o):o.zero_grad();loss=objective(m,[row],'cpu');loss.backward();o.step();return loss
 assert torch.isfinite(step(p.model,opt));ck=tmp_path/'resume.pt';torch.save(dict(model=p.model.state_dict(),optimizer=opt.state_dict()),ck)
 clone=copy.deepcopy(p.model);oo=torch.optim.Adam([x for x in clone.parameters() if x.requires_grad],lr=1e-4);d=torch.load(ck,weights_only=True);clone.load_state_dict(d['model']);oo.load_state_dict(d['optimizer']);step(p.model,opt);step(clone,oo)
 assert all(torch.equal(t,clone.state_dict()[k]) for k,t in p.model.state_dict().items())
 assert all(torch.equal(t,before[k]) for k,t in p.model.state_dict().items() if not k.startswith('policy.'))
 assert any(not torch.equal(t,before[k]) for k,t in p.model.state_dict().items() if k.startswith('policy.'))
 with torch.no_grad():_,newv=p.model(*collate([obs],'cpu'))
 assert torch.equal(oldv,newv) and v(s)==vv
