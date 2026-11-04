"""Bounded training with full resumable state and explicit teacher provenance."""
import json
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
from .engine import State, legal_moves, play
from .ai import heuristic, seed_all
from .research import PolicyValue, Agent, pack, to_device, search, checkpoint, restore, source_hashes, sha


def train(method, output, baseline, seconds=180, games=100000, seed=101, device='cuda', resume=None, lr=.0005, width=4, reply_width=3):
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    if (out/'latest.pt').exists() and resume is None: raise ValueError('output already trained; use --resume or a new directory')
    seed_all(seed)
    model = PolicyValue(method).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    teacher = Agent(str(baseline), device)
    self_agent = Agent('heuristic', device); self_agent.kind = 'research'; self_agent.model = model
    pool = [Agent('random', device), Agent('heuristic', device), teacher, self_agent]
    episode = 0; previous_seconds = 0.; total_nodes = 0; steps = 0; previous_metrics = {}
    if resume:
        ck = restore(resume, model, opt, rng=True); episode = ck['episode']; previous_metrics = ck['metadata']
        previous_seconds = ck['metadata']['training_seconds']; total_nodes = ck['metadata']['search_nodes']; steps = ck['metadata']['learner_positions']
    config = dict(method=method, output=str(output), baseline=str(baseline), baseline_sha256=sha(baseline), seconds=seconds, games=games,
                  seed=seed, device=device, resume=str(resume) if resume else None, lr=opt.param_groups[0]['lr'], width=width, reply_width=reply_width,
                  torch=torch.__version__, cuda=torch.version.cuda, python=sys.version, source_hashes=source_hashes(),
                  algorithm='beam-search distillation, student-guided root shortlist' if method=='C' else 'v1+heuristic ranking distillation with on-policy actor-critic auxiliary',
                  pool=['random','heuristic','v1','current self'], prior='fixed v1 heuristic logits + trainable residual')
    (out/'config.json').write_text(json.dumps(config, indent=2))
    with (out/'config-history.jsonl').open('a') as f: f.write(json.dumps(config)+'\n')
    start = time.perf_counter(); initial_episode = episode; saved_midpoint = False
    metrics = dict(rollout_seconds=previous_metrics.get('rollout_seconds',0.), update_seconds=previous_metrics.get('update_seconds',0.), search_nodes=total_nodes, learner_positions=steps, training_seconds=previous_seconds, config=config)
    if resume is None: checkpoint(out/'untrained.pt', model, opt, 0, metrics.copy())
    with (out/'train.jsonl').open('a') as log:
        while episode-initial_episode < games and time.perf_counter()-start < seconds:
            t = time.perf_counter(); s = State(); trajectory = []
            # Training starts use the training RNG, never the evaluation prefix files.
            for _ in range(random.randint(2,6)):
                qs = legal_moves(s); play(s, random.choice(qs).move if qs else None, validate=False)
            seat = random.randrange(2); opponent_index = random.randrange(4); opponent = pool[opponent_index]
            while not s.done:
                if s.turn != seat:
                    play(s, opponent.act(s, random), validate=False); continue
                qs = legal_moves(s)
                if not qs: play(s, None, validate=False); continue
                data = pack(s, qs)
                with torch.no_grad():
                    logits, _ = model(*to_device(data, device))
                    if method == 'C':
                        _, selected, values, nodes = search(s, logits.cpu().numpy(), width, reply_width)
                        total_nodes += nodes
                        # Unsearched candidates receive zero target mass; no claim they were evaluated.
                        target = np.zeros(len(qs), dtype=np.float32)
                        qvalues = torch.as_tensor(values*40, device=device)
                        target[selected] = qvalues.softmax(0).cpu().numpy()
                        action = int(random.choices(range(len(qs)), weights=target, k=1)[0])
                    else:
                        x, st = data[2], data[1][:46]
                        v1_logits, _ = teacher.model(torch.as_tensor(x, device=device), torch.as_tensor(st, device=device))
                        v1_std = v1_logits.std(unbiased=False).clamp_min(.05)
                        teacher_logits = torch.as_tensor(heuristic(x), device=device)+2*(v1_logits-v1_logits.mean())/v1_std
                        target = teacher_logits.softmax(0).cpu().numpy()
                        action = int(torch.distributions.Categorical(logits=logits).sample())
                trajectory.append((data, target, action, s.turn))
                play(s, qs[action].move, validate=False)
            rollout_seconds = time.perf_counter()-t; t = time.perf_counter(); opt.zero_grad(); losses = []
            for data, target, action, perspective in trajectory:
                logits, value = model(*to_device(data, device)); distribution = torch.distributions.Categorical(logits=logits)
                reward = (s.score(perspective)-s.score(1-perspective))/109
                target_tensor = torch.as_tensor(target, device=device)
                imitation = -(target_tensor*logits.log_softmax(0)).sum()
                policy_gradient = -distribution.log_prob(torch.tensor(action, device=device))*(reward-value.detach())
                # C actions came from the expert, so no invalid on-policy gradient is applied.
                loss = imitation+.5*(value-reward).square()
                if method != 'C': loss = loss+.15*policy_gradient-.002*distribution.entropy()
                (loss/max(1,len(trajectory))).backward(); losses.append(loss.detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            if device.startswith('cuda'): torch.cuda.synchronize()
            update_seconds = time.perf_counter()-t; episode += 1; steps += len(trajectory)
            metrics.update(training_seconds=previous_seconds+time.perf_counter()-start, search_nodes=total_nodes, learner_positions=steps)
            metrics['rollout_seconds'] += rollout_seconds; metrics['update_seconds'] += update_seconds
            record = dict(episode=episode, updates=episode, opponent=config['pool'][opponent_index], scores=[s.score(0),s.score(1)],
                          loss=float(torch.stack(losses).mean()) if losses else 0, training_seconds=metrics['training_seconds'],
                          rollout_seconds=rollout_seconds, update_seconds=update_seconds, search_nodes=total_nodes, learner_positions=steps)
            log.write(json.dumps(record)+'\n'); log.flush()
            if episode % 25 == 0: print(json.dumps(record), flush=True)
            if not saved_midpoint and time.perf_counter()-start >= seconds/2:
                checkpoint(out/'midpoint.pt', model, opt, episode, metrics.copy()); saved_midpoint = True
            if episode % 25 == 0: checkpoint(out/'latest.pt', model, opt, episode, metrics.copy())
        metrics['training_seconds'] = previous_seconds+time.perf_counter()-start
        checkpoint(out/'latest.pt', model, opt, episode, metrics.copy())
        if not saved_midpoint: checkpoint(out/'midpoint.pt', model, opt, episode, metrics.copy())
    result = dict(method=method, completed_games=episode-initial_episode, total_games=episode, updates=episode,
                  invocation_seconds=time.perf_counter()-start, **metrics)
    (out/'summary.json').write_text(json.dumps(result, indent=2)); print(json.dumps({k:v for k,v in result.items() if k!='config'}), flush=True)
    return result
