import random
from collections import Counter
import pytest
from blokus.engine import *

def test_pieces_and_orientations():
    assert Counter(SIZES)=={1:1,2:1,3:2,4:5,5:12}
    assert sum(SIZES)==89 and len(PIECES)==21
    assert len({min(orientations(p)) for p in PIECES})==21
    assert sum(map(len,ORIENTS))==91
    for p,os in zip(PIECES,ORIENTS):
        assert len(os)==len(set(os))
        for q in os:assert orientations(q)==os

def test_opening_and_turns():
    s=State();assert not is_legal(s,None)
    assert not is_legal(s,Move(0,0,0,0))
    play(s,Move(0,0,4,4));assert s.turn==1
    assert not is_legal(s,Move(0,0,4,5))
    play(s,Move(0,0,9,9));assert s.turn==0
    assert not is_legal(s,Move(0,0,5,5)) # consumed mono
    assert not is_legal(s,Move(1,0,-1,5))

def test_contact_and_overlap():
    s=State(boards=[bit((4,4)),bit((5,6))])
    assert is_legal(s,Move(0,0,5,5)) # own diagonal + opponent edge
    assert not is_legal(s,Move(0,0,4,5))
    assert not is_legal(s,Move(0,0,4,4))
    assert not is_legal(s,Move(0,0,5,6))
    assert not is_legal(s,Move(0,0,7,7))
    assert not is_legal(s,Move(0,0,14,14))

def test_pass_terminal_and_score():
    s=State(remaining=[0,1],last=[1,None]);assert is_legal(s,None)
    play(s,None);assert not s.done
    play(s,Move(0,0,9,9));assert s.passes==0
    play(s,None);play(s,None);assert s.done
    assert [s.score(0),s.score(1)]==[15,20]
    assert not is_legal(s,None)
    with pytest.raises(ValueError):play(s,None)
    s=State();assert s.score(0)==-89
    s.remaining[0]=(1<<1)|(1<<2);assert s.score(0)==-5

def test_exhaustive_enumeration_matches_independent_checker():
    rng=random.Random(123);s=State()
    for turn in range(40):
        qs=legal_moves(s)
        assert len({q.move for q in qs})==len(qs)
        assert all(is_legal(s,q.move) for q in qs)
        if turn in (0,1,10,20,30,39):
            expected={q.move for q in PLACEMENTS if is_legal(s,q.move)}
            assert {q.move for q in qs}==expected
        if s.done:break
        play(s,rng.choice(qs).move if qs else None)

def test_random_games_conserve_cells_and_terminate():
    for seed in range(5):
        s=State();rng=random.Random(seed);turns=0
        while not s.done:
            qs=legal_moves(s);play(s,rng.choice(qs).move if qs else None);turns+=1
            assert not s.boards[0]&s.boards[1]
            for p in (0,1):assert s.boards[p].bit_count()+sum(SIZES[i] for i in range(21) if s.remaining[p]>>i&1)==89
            assert turns<=86
        assert not legal_moves(s)

def test_edge_contact_overrides_simultaneous_corner_contact():
    s=State(boards=[mask({(4,4),(5,4)}),0])
    assert not is_legal(s,Move(0,0,5,5))
    assert all(q.move!=Move(0,0,5,5) for q in legal_moves(s))

def test_opening_count_and_rotation_symmetry():
    assert len(legal_moves(State()))==414
    s=State();rng=random.Random(8)
    for _ in range(12):
        qs=legal_moves(s);play(s,rng.choice(qs).move)
    # Rotate 180 degrees and exchange colours/start points.
    rot=lambda cells:tuple(sorted((13-r,13-c) for r,c in cells))
    other=State(boards=[mask(rot(cells_of(s.boards[1]))),mask(rot(cells_of(s.boards[0])))],
                remaining=s.remaining[::-1],last=s.last[::-1],turn=1-s.turn,passes=s.passes)
    a={(q.move.piece,rot(q.move.cells)) for q in legal_moves(s)}
    b={(q.move.piece,tuple(sorted(q.move.cells))) for q in legal_moves(other)}
    assert a==b
