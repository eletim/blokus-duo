"""Read-only observation server: no inference, training or archive writes."""
import argparse
from datetime import datetime, timezone
from functools import lru_cache
import gzip
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sqlite3
from urllib.parse import urlsplit, parse_qs

from .engine import State, Move, play, cells_of, PIECES


def read_json(path):
    path = Path(path)
    return cached_json(str(path), path.stat().st_mtime_ns, path.stat().st_size)


@lru_cache(maxsize=64)
def cached_json(path, modified, size):
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt') as handle:
        return json.load(handle)


def frames(moves, expected_scores=None):
    state = State()
    def snapshot():
        return dict(boards=[list(cells_of(b)) for b in state.boards],
                    remaining=[[i for i in range(21) if state.remaining[p] >> i & 1] for p in (0, 1)],
                    scores=[state.score(p) for p in (0, 1)], turn=state.turn, done=state.done)
    result = [snapshot()]
    for raw in moves:
        play(state, Move(*raw) if raw is not None else None, validate=True)
        result.append(snapshot())
    if expected_scores is not None and result[-1]['scores'] != expected_scores:
        raise ValueError('棋譜の再生結果と保存スコアが一致しません')
    return result


class Sources:
    def __init__(self, artifacts, archive):
        self.artifacts = Path(artifacts).resolve()
        self.archive = Path(archive).resolve()
        self.stages = [(24576, self.artifacts/'value-placement-24576'),
                       (36864, self.artifacts/'value-placement-49152/n36864'),
                       (49152, self.artifacts/'value-placement-49152/n49152')]

    @staticmethod
    def model(code, stage):
        return {'A': f'配置価値A / {stage:,}局' + (' (gen4)' if stage == 49152 else ''),
                'F': 'gen3 / Frozen', 'Q': '配置価値A / 3,072局',
                'R': '配置価値A / 24,576局',
                'P': f'配置価値A / {24576 if stage == 36864 else 36864:,}局',
                'C': f'配置価値C / {stage:,}局'}.get(code, code)

    def shards(self):
        return [(n, p) for n, root in self.stages for phase in ('confirmation', 'validation')
                for p in sorted(root.glob(phase+'-??-*.json.gz'))]

    def catalog(self):
        groups, evaluations, errors, training = [], [], [], []
        for stage, path in self.shards():
            try:
                data = read_json(path)
                groups.append(dict(id=str(path.relative_to(self.artifacts)), stage=stage,
                    phase=path.name.split('-')[0], seed=data['seed'], count=len(data['games']),
                    left=self.model(data['left'], stage), right=self.model(data['right'], stage)))
            except (OSError, ValueError, KeyError) as error:
                errors.append(f'{path.name}: {error}')
        for stage, root in self.stages:
            for phase in ('validation', 'confirmation'):
                path = root/f'{phase}-summary.json'
                if not path.exists():
                    continue
                try:
                    for pair, row in read_json(path).items():
                        n = row['games']
                        if n <= 0 or row['wins'] + row['draws'] + row['losses'] != n:
                            raise ValueError('評価の対局数が不一致')
                        evaluations.append(dict(stage=stage, phase=phase,
                            opponent=self.model(pair[1], stage), pair=pair, games=n,
                            wins=row['wins'], draws=row['draws'], losses=row['losses'],
                            win_rate=row['wins']/n, point=row['point'],
                            source=str(path.relative_to(self.artifacts))))
                except (OSError, ValueError, KeyError) as error:
                    errors.append(f'{path.name}: {error}')
            path = root/'learning-A-1751401.json'
            if path.exists():
                try:
                    d = read_json(path); curve = d['curve']; cfg = d['config']
                    selection = root/'checkpoint-selection.json'
                    training.append(dict(stage=stage, seed=1751401,
                        games=cfg.get('train_games'), epochs=curve[-1]['epoch'] if curve else None,
                        target_epochs=cfg.get('epochs'), updates=d.get('updates'),
                        selected_epoch=read_json(selection).get('epoch') if selection.exists() else None,
                        source=str(path.relative_to(self.artifacts))))
                except (OSError, ValueError, KeyError) as error:
                    errors.append(f'{path.name}: {error}')
        root = self.artifacts/'value-placement-49152'
        progress = dict(generated=None, finished=None, phases=[], source='value-placement-49152')
        for name in ('games/timing.json', 'pipeline-complete.json', 'phase-times.json'):
            path = root/name
            if not path.exists():
                continue
            try:
                d = read_json(path)
                if name == 'games/timing.json': progress['generated'] = d.get('games')
                elif name == 'pipeline-complete.json': progress['finished'] = d.get('finished_utc') if d.get('complete') else None
                else: progress['phases'] = d
            except (OSError, ValueError) as error:
                errors.append(f'{name}: {error}')
        archived = []
        if self.archive.exists():
            try:
                with sqlite3.connect(self.archive.as_uri()+'?mode=ro', uri=True) as db:
                    for key, started, mode, raw in db.execute("SELECT id,started,mode,record FROM games WHERE kind='human' ORDER BY started DESC"):
                        r = json.loads(raw)
                        archived.append(dict(id=key, started=started, mode=mode, status=r['status'], plies=len(r['moves'])))
            except (sqlite3.Error, ValueError, KeyError) as error:
                errors.append(f'人間対AI棋譜: {error}')
        return dict(groups=groups, evaluations=evaluations, training=training, progress=progress,
                    archive=archived, errors=errors, pieces=PIECES,
                    updated=datetime.now(timezone.utc).isoformat())

    def game(self, group, index):
        # Only indexed evaluation shards can be opened; no arbitrary paths.
        selected = next(((stage, path) for stage, path in self.shards()
                         if str(path.relative_to(self.artifacts)) == group), None)
        if selected is None:
            raise KeyError('対戦データが見つかりません')
        stage, path = selected
        data = read_json(path)
        if not 0 <= index < len(data['games']):
            raise KeyError('対局番号が範囲外です')
        game = data['games'][index]
        phase = path.name.split('-')[0]
        prefix = read_json(path.parent/f'{phase}-bank.json')['prefixes'][game['prefix']]['moves']
        moves = prefix + game['moves']
        players = [None, None]
        players[game['seat']] = self.model(data['left'], stage)
        players[1-game['seat']] = self.model(data['right'], stage)
        return dict(players=players, seed=data['seed'], prefix_plies=len(prefix), moves=moves,
                    frames=frames(moves, game['scores']), source=group, index=index,
                    count=len(data['games']), kind='AI評価対戦', budget_ms=data['budget_ms'])

    def human_game(self, key):
        if not self.archive.exists(): raise KeyError('棋譜は未取得です')
        with sqlite3.connect(self.archive.as_uri()+'?mode=ro', uri=True) as db:
            row = db.execute("SELECT record FROM games WHERE id=? AND kind='human'", (key,)).fetchone()
        if row is None: raise KeyError('棋譜が見つかりません')
        r = json.loads(row[0]); players = [r['generation'], r['generation']]
        players[r['human_seat']] = '人間'
        moves = [m['move'] for m in r['moves']]
        return dict(players=players, seed=r.get('model_seed'), prefix_plies=0, moves=moves,
                    frames=frames(moves, r['scores']), source='人間対AI 保存棋譜', index=0,
                    count=1, kind='人間対AI', budget_ms=r.get('budget_ms'))


def handler(sources):
    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, data, content_type='application/json; charset=utf-8'):
            body = data if isinstance(data, bytes) else json.dumps(data, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            url = urlsplit(self.path); query = parse_qs(url.query)
            try:
                if url.path == '/':
                    self.respond(200, Path(__file__).with_name('console.html').read_bytes(), 'text/html; charset=utf-8')
                elif url.path == '/api/catalog': self.respond(200, sources.catalog())
                elif url.path == '/api/replay':
                    self.respond(200, sources.game(query['group'][0], int(query.get('index', ['0'])[0])))
                elif url.path == '/api/archive': self.respond(200, sources.human_game(query['id'][0]))
                else: self.respond(404, dict(error='not found'))
            except KeyError as error: self.respond(404, dict(error=str(error)))
            except (ValueError, IndexError, AssertionError) as error: self.respond(422, dict(error=str(error) or '棋譜検証エラー'))
            except (OSError, sqlite3.Error) as error: self.respond(503, dict(error='データを取得できません: '+str(error)))
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--archive', type=Path, default=Path.home()/'.local/share/blokus-web/games.sqlite3')
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), handler(Sources(args.artifacts, args.archive)))
    print(f'Console: http://{args.host}:{args.port}', flush=True)
    try: server.serve_forever()
    finally: server.server_close()


if __name__ == '__main__': main()
