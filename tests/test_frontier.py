"""Compare bit shifts with a coordinate oracle, including every board boundary."""
import random
from blokus.engine import State, N, START, ORTH, DIAG, bit, cells_of, frontier


def coordinate_frontier(boards, player):
    own = set(cells_of(boards[player]))
    occupied = set(cells_of(boards[0] | boards[1]))
    def adjacent(directions):
        return {(x+dx, y+dy) for x, y in own for dx, dy in directions
                if 0 <= x+dx < N and 0 <= y+dy < N}
    edge = adjacent(ORTH)
    corners = adjacent(DIAG) if own else {START[player]}
    encode = lambda cells: sum(bit(c) for c in cells)
    return encode(corners - occupied - edge), encode(edge)


def test_frontier_board_edges_and_empty_start():
    for player in range(2):
        assert frontier(State(), player) == coordinate_frontier([0, 0], player)
        for r in range(N):
            for c in range(N):
                boards = [0, 0]
                boards[player] = bit((r, c))
                assert frontier(State(boards=boards), player) == coordinate_frontier(boards, player)


def test_frontier_random_dense_and_sparse_boards():
    rng = random.Random(724503)
    for density in (0.05, 0.3, 0.9):
        for _ in range(100):
            boards = [0, 0]
            for r in range(N):
                for c in range(N):
                    if rng.random() < density:
                        boards[rng.randrange(2)] |= bit((r, c))
            for player in range(2):
                assert frontier(State(boards=boards), player) == coordinate_frontier(boards, player)
