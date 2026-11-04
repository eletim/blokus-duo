"""Independent, deliberately simple coordinate-set oracle for engine review."""
from collections import Counter

SIZE = 14
STARTS = ((4, 4), (9, 9))
ORTH = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIAG = ((1, 1), (1, -1), (-1, 1), (-1, -1))

def normalize(cells):
    cells = tuple(cells)
    x0, y0 = min(x for x, y in cells), min(y for x, y in cells)
    return tuple(sorted((x-x0, y-y0) for x, y in cells))

def orientations(cells):
    result = set()
    for flip in (1, -1):
        shape = [(flip*x, y) for x, y in cells]
        for _ in range(4):
            result.add(normalize(shape))
            shape = [(-y, x) for x, y in shape]
    return result

def pieces():
    current = {((0, 0),)}
    result = set(current)
    for size in range(2, 6):
        grown = set()
        for shape in current:
            for x, y in shape:
                for dx, dy in ORTH:
                    p = (x+dx, y+dy)
                    if p not in shape:
                        grown.add(min(orientations((*shape, p))))
        result |= grown
        current = grown
    return result

def valid(cells, own, opponent, start):
    cells, own, opponent = set(cells), set(own), set(opponent)
    if not cells or any(not (0 <= x < SIZE and 0 <= y < SIZE) for x, y in cells):
        return False
    if cells & (own | opponent):
        return False
    if not own:
        return start in cells
    if any((x+dx, y+dy) in own for x, y in cells for dx, dy in ORTH):
        return False
    return any((x+dx, y+dy) in own for x, y in cells for dx, dy in DIAG)

def legal_cells(shape, own, opponent, start):
    found = set()
    for orient in orientations(shape):
        for x in range(SIZE-max(a for a, b in orient)):
            for y in range(SIZE-max(b for a, b in orient)):
                cells = tuple(sorted((x+a, y+b) for a, b in orient))
                if valid(cells, own, opponent, start):
                    found.add(cells)
    return found

if __name__ == '__main__':
    shapes = pieces()
    counts = Counter(map(len, shapes))
    assert counts == {1: 1, 2: 1, 3: 2, 4: 5, 5: 12}, counts
    assert sum(len(s) for s in shapes) == 89
    opening = sum(len(legal_cells(s, (), (), STARTS[0])) for s in shapes)
    print({'piece_counts': dict(counts), 'area': 89, 'opening_legal_moves': opening})
