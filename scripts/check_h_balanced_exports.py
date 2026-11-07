from h_balanced import *
from blokus.ppo_value_leaf import value_only
base.setup();checks=[]
for seed in SEEDS:
 source=torch.load(f'artifacts/ppo-gen3/seed-{seed}.pt',map_location='cpu',weights_only=True)
 for arm in ('A','B'):
  ck=torch.load(R/f'{arm}-{seed}.pt',map_location='cpu',weights_only=True);assert ck['updates']==96 and ck['epochs']==8
  assert ck['teacher_sha256']==sha(f'artifacts/ppo-gen3/seed-{seed}.pt') and ck['training_data_sha256']==sha(R/f'{arm}-train-{seed}.json')
  assert all(torch.equal(v,ck['model'][k]) for k,v in source['model'].items() if not k.startswith('policy.'))
  rows=json.loads((R/f'balanced-dev-{seed}.json').read_text())['labels'][::17]
  for device in ('cpu','cuda'):
   a=PPOAgent(seed,device);b=PPOAgent(seed,device,checkpoint=R/f'{arm}-{seed}.pt')
   for r in rows:assert value_only(a,state(r['moves']))==value_only(b,state(r['moves']))
  checks.append(dict(seed=seed,arm=arm,sha256=sha(R/f'{arm}-{seed}.pt'),updates=96,frozen_values_cpu_cuda_within_device_bit_equal=True))
write(R/'pre-evaluation-exports.json',dict(pass_=True,checks=checks))
