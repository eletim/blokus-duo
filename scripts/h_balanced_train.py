from h_balanced import *
from h_distill_train import cache,Trainer,metrics

def run():
 assert (R/'pre-training-counts.json').exists()
 for seed in SEEDS:
  p=PPOAgent(seed,'cpu');devrows=json.loads((R/f'balanced-dev-{seed}.json').read_text())['labels'];dev=cache(p,devrows)
  for arm in ('A','B'):
   started=time.perf_counter();rows=json.loads((R/f'{arm}-train-{seed}.json').read_text())['labels'];data=cache(p,rows);t=Trainer(seed,data);logs=[]
   assert len(data)==192
   before=metrics(t.head,dev,'cuda')
   for ep in range(8):
    us=[t.step() for _ in range(12)];torch.cuda.synchronize()
    logs.append(dict(epoch=ep+1,updates=t.updates,train=metrics(t.head,data,'cuda'),dev=metrics(t.head,dev,'cuda'),last_update=us[-1],wall_seconds=time.perf_counter()-started));print(arm,seed,ep+1,logs[-1]['dev'],flush=True)
    t.save(R/f'resume-{arm}-{seed}.pt');write(R/f'curve-{arm}-{seed}.json',logs)
   src=torch.load(f'artifacts/ppo-gen3/seed-{seed}.pt',map_location='cpu',weights_only=True)
   model=copy.deepcopy(src['model'])
   for k,v in t.head.state_dict().items():model['policy.'+k]=v.cpu()
   changed=[k for k,v in model.items() if not torch.equal(v,src['model'][k])];assert changed and all(k.startswith('policy.') for k in changed)
   torch.save(dict(format='terminal-ppo-inference-v2',channels=4,model=model,research_name='H-balanced-control-'+arm,teacher_sha256=sha(f'artifacts/ppo-gen3/seed-{seed}.pt'),seed=seed,epochs=8,updates=t.updates,training_data_sha256=sha(R/f'{arm}-train-{seed}.json')),R/f'{arm}-{seed}.pt')
   write(R/f'learning-{arm}-{seed}.json',dict(before_dev=before,curve=logs,changed_parameters=changed,wall_seconds=time.perf_counter()-started,device=torch.cuda.get_device_name(),torch_version=torch.__version__))
if __name__=='__main__':base.setup();run()
