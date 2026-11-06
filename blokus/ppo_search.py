"""Budgeted beam negamax for human play. Strength is experimental."""
import copy
import time
import torch
from .engine import legal_moves, play
from .ppo_policy import collate
from .ppo_value_leaf import value_only


class SearchAgent:
    def __init__(self, policy, budget_ms=500, width=4, max_depth=4):
        if not 10 <= budget_ms <= 5000 or not 1 <= width <= 16 or not 1 <= max_depth <= 8:
            raise ValueError('invalid search limits')
        self.policy, self.budget_ms = policy, budget_ms
        self.width, self.max_depth = width, max_depth
        self.stats = {}

    def act(self, state, rng):
        start = time.perf_counter()
        deadline = start + self.budget_ms / 1000
        nodes = 0

        def evaluate(s, leaf=False):
            nonlocal nodes
            if time.perf_counter() >= deadline:
                raise TimeoutError
            nodes += 1
            if s.done:
                difference = s.score(s.turn) - s.score(1-s.turn)
                return [], float((difference > 0) - (difference < 0))
            if leaf:
                value = value_only(self.policy, s)
                if time.perf_counter() >= deadline:
                    raise TimeoutError
                return [], value
            obs, qs = self.policy._observation(s)
            with torch.inference_mode():
                logits, value = self.policy.model(*collate([obs], self.policy.device))
                order = logits[0].argsort(descending=True).cpu().tolist()
                v = value.item()
            if time.perf_counter() >= deadline:
                raise TimeoutError
            return ([qs[i].move for i in order[:self.width]] if qs else [None]), v

        def visit(s, depth):
            moves, value = evaluate(s, leaf=not depth)
            if not depth or s.done:
                return value
            best = -float('inf')
            for move in moves:
                child = copy.deepcopy(s)
                play(child, move, validate=False)
                best = max(best, -visit(child, depth-1))
            return best

        # Always have a legal policy move, even if a single inference exceeds budget.
        selected = self.policy.act(state, rng)
        completed = 0
        if not legal_moves(state):
            self.stats = dict(depth=0, nodes=0, elapsed_ms=(time.perf_counter()-start)*1000, fallback=True)
            return None
        try:
            roots, _ = evaluate(state)
            # Always retain the baseline in the root beam (ties can differ).
            if selected not in roots:
                roots[-1] = selected
            for depth in range(2, self.max_depth+1, 2):
                scores = []
                for move in roots:
                    child = copy.deepcopy(state)
                    play(child, move, validate=False)
                    scores.append(-visit(child, depth-1))
                # Commit only after all root candidates finish at the same depth.
                selected = roots[max(range(len(roots)), key=lambda i: scores[i])]
                completed = depth
        except TimeoutError:
            pass
        self.stats = dict(depth=completed, nodes=nodes, elapsed_ms=(time.perf_counter()-start)*1000, fallback=not completed)
        return selected
