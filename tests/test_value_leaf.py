"""Completed-tree equivalence; time-limited moves may change with faster leaves."""
import copy
import random
import numpy as np
import torch
from blokus.engine import State, play
from blokus.ppo_policy import PPOAgent, collate
from blokus.ppo_search import SearchAgent
from blokus.ppo_value_leaf import value_inputs, value_only


def full_value(policy, state):
    if state.done:
        d = state.score(state.turn) - state.score(1-state.turn)
        return float(np.sign(d))
    obs, _ = policy._observation(state)
    with torch.inference_mode():
        return policy.model(*collate([obs], policy.device))[1].item()


def test_leaf_inputs_and_values():
    torch.set_num_threads(1)
    for generation, seed in [("gen3", 1751401), ("gen2", 1401)]:
        policy = PPOAgent(seed, "cpu", generation=generation)
        state = State()
        states = [copy.deepcopy(state)]
        for ply in range(32):
            if state.done:
                break
            play(state, policy.act(state, random.Random(ply)), validate=False)
            if ply % 4 == 0:
                states.append(copy.deepcopy(state))
        states += [State(remaining=[1, 0], last=[None, 1], turn=1),
                   State(remaining=[0, 0], last=[0, 1], passes=2)]
        for state in states:
            if not state.done:
                obs, _ = policy._observation(state)
                board, features = value_inputs(state, policy.channels)
                assert np.array_equal(board, obs[0])
                assert np.array_equal(features, obs[1])
            assert abs(value_only(policy, state)-full_value(policy, state)) <= 1e-7


def test_leaf_skips_candidates(monkeypatch):
    policy = PPOAgent(1751401, "cpu")
    expected = full_value(policy, State())
    def forbidden(*args):
        raise AssertionError("leaf must not create policy candidates")
    monkeypatch.setattr(policy, "_observation", forbidden)
    assert value_only(policy, State()) == expected


def test_completed_tree_matches_full_value(monkeypatch):
    import blokus.ppo_search as search
    torch.set_num_threads(1)
    policy = PPOAgent(1751401, "cpu")
    for state in [State(), State(remaining=[1, 1]),
                  State(remaining=[1, 0], last=[None, 1], turn=1)]:
        fast = SearchAgent(policy, 5000, width=2)
        expected = fast.act(state, random.Random(10))
        with monkeypatch.context() as m:
            m.setattr(search, "value_only", full_value)
            reference = SearchAgent(policy, 5000, width=2)
            assert reference.act(state, random.Random(10)) == expected
            assert reference.stats["depth"] == fast.stats["depth"]
