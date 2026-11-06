"""Finite-run model, terminal reward, source, resume and benchmark audits."""
import hashlib,json,subprocess
from pathlib import Path
import numpy as np
import torch
from blokus import ppo_runtime as r
from blokus.ppo_policy import PPOAgent,observe,collate
from blokus.engine import State
from blokus.value_experiment import write
from scripts.bench_ppo_runtime import audit_games,equal
ROOT=Path('artifacts/ppo-four-channel-runtime')
bench=json.loads((ROOT/'benchmark.json').read_text());assert bench['all_bit_equal'] and all(x['bit_equal'] for x in bench['equality'])
for row in bench['rows']:
 for key,number in [('games_per_second',16),('learner_states_per_second',row['learner_states']),('optimizer_steps_per_second',row['optimizer_steps'])]:assert abs(row[key]-number/row['wall'])<1e-9
for channels in [4,6]:
 for repeat in range(3):
  rows=[x for x in bench['rows'] if x['channels']==channels and x['repeat']==repeat];assert len({x['model_hash'] for x in rows})==1 and len({x['input_hash'] for x in rows})==1 and len({x['optimizer_steps'] for x in rows})==1
initial=torch.load(ROOT/'pilot/untrained.pt',map_location='cpu',weights_only=False);final=torch.load(ROOT/'pilot/latest.pt',map_location='cpu',weights_only=False)
assert final['channels']==4 and final['iteration']==64 and final['workers'] is None and final['model']['conv.0.weight'].shape[1]==4
assert any(not torch.equal(initial['model'][k],v) for k,v in final['model'].items())
for log in final['logs']:
 assert all(np.isfinite(v) for v in log['update'].values()) and log['update']['grad_norm']>0
rows=[json.loads(line) for line in (ROOT/'pilot/games.jsonl').read_text().splitlines()];assert [x['iteration'] for x in rows]==list(range(1,65));games=[]
for row,log in zip(rows,final['logs']):
 assert hashlib.sha256(json.dumps(row['games'],sort_keys=True).encode()).hexdigest()==log['games_sha256'];games.extend(row['games'])
audit_games(games);assert len(games)==1024
agent=PPOAgent(device='cpu',checkpoint=ROOT/'pilot/inference.pt');assert equal(agent.model.state_dict(),final['model'])
old=r.TerminalNet(4);old.load_state_dict(initial['model']);obs=observe(State(),channels=4)[0]
with torch.no_grad():
 old_logits,_=old(*collate([obs],'cpu'));new_logits,_=agent.model(*collate([obs],'cpu'));p=old_logits.softmax(-1);kl=float((p*(old_logits.log_softmax(-1)-new_logits.log_softmax(-1))).sum())
assert kl>0
assert Path('blokus/engine.py').read_bytes()==subprocess.check_output(['git','show','a011c78:blokus/engine.py'])
for d in ['artifacts/ppo-gen2','artifacts/ppo-2048']:assert not subprocess.check_output(['git','diff','c901d09','--',d])
write(ROOT/'pilot-audit.json',dict(pass_=True,channels=4,games=1024,iterations=64,optimizer_steps=sum(x['update']['updates'] for x in final['logs']),model_changed=True,initial_policy_KL=kl,grad_norm_range=[min(x['update']['grad_norm'] for x in final['logs']),max(x['update']['grad_norm'] for x in final['logs'])],exports_exact=True,all_terminal_legal_learner_side=True,engine_and_public_models_unchanged=True,worker_state=None,not_strength_evaluation=True))
print('pilot/benchmark audit PASS',len(games),kl)
