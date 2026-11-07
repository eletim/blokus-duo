"""Descriptive post-evaluation diagnostics only; no selection/extra training."""
from h_distill_train import *
setup();out={}
for seed in SEEDS:
 p=PPOAgent(seed,'cpu');q=PPOAgent(seed,'cpu',checkpoint=R/f'student-{seed}.pt');rows=json.loads((R/f'dev-{seed}.json').read_text())['labels'];changed=0;lost=0;topoverlap=[];ranks=[];turns={0:0,1:0}
 for row in rows:
  s=state(row['moves']);turns[s.turn]+=1;obs,qs=p._observation(s)
  with torch.no_grad():a,_=p.model(*collate([obs],'cpu'));b,_=q.model(*collate([obs],'cpu'))
  ao=a[0].argsort(descending=True).tolist();bo=b[0].argsort(descending=True).tolist();target=[raw(x.move) for x in qs].index(row['target'])
  changed+=ao[0]!=bo[0];lost+=target not in bo[:4];topoverlap.append(len(set(ao[:4])&set(bo[:4]))/min(4,len(ao)));ranks.append([ao.index(target)+1,bo.index(target)+1])
 # Cached CPU feature head CUDA agrees with CPU inference within predefined tolerance.
 data=cache(q,rows[:8]);head=copy.deepcopy(q.model.policy).cuda();maxerr=0
 for item in data:
  with torch.no_grad():gpu=head(item[0].cuda()).squeeze(-1).cpu()+item[1];cpu=q.model.policy(item[0]).squeeze(-1)+item[1]
  torch.testing.assert_close(gpu,cpu,atol=2e-6,rtol=1e-6);maxerr=max(maxerr,float((gpu-cpu).abs().max()))
 out[str(seed)]=dict(dev_roots=len(rows),root_turns=turns,greedy_changes=changed,teacher_chosen_dropped_from_student_top4=lost,mean_top4_overlap=float(np.mean(topoverlap)),mean_teacher_rank=np.mean(ranks,axis=0).tolist(),cached_cuda_cpu_logit_max_absolute_difference=maxerr)
byseat={};fallback={}
from h_distill_eval import summary
for method in ('H','policy'):
 games=[g for seed in SEEDS for g in json.loads((R/f'validation-{method}-{seed}.json').read_text())['games']];byseat[method]={str(seat):summary([g for g in games if g['seat']==seat]) for seat in (0,1)}
 if method=='H':
  for side in (0,1):
   ts=[t for g in games for t in g['timings'] if t['side']==side];fallback[str(side)]=dict(forced_pass=sum(t.get('forced_pass',False) for t in ts),nonpass_fallback=sum(t.get('fallback',False) and not t.get('forced_pass',False) for t in ts))
write(R/'final-diagnostic.json',dict(dev_by_seed=out,validation_by_seat=byseat,H_fallback=fallback,posthoc_descriptive_only=True))
