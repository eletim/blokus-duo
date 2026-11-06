"""Local browser game: python -m blokus.web_play."""
import argparse
import copy
import json
import random
import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from .engine import State, Move, ORIENTS, START, cells_of, legal_moves, play
from .ppo_search import SearchAgent


class Games:
    def __init__(self, bot, linear_factory=None):
        self.bot = bot
        self.linear_factory = linear_factory
        self.games = {}
        self.lock = threading.RLock()
        self.worker = ThreadPoolExecutor(max_workers=1)

    def create(self, seat, mode, budget):
        if type(seat) is not int or seat not in (0, 1) or mode not in ('policy', 'search', 'linear') or type(budget) is not int or budget not in (100, 500, 2000):
            raise ValueError('対局設定が不正です')
        if mode == 'linear' and self.linear_factory is None:
            raise ValueError('非NN AIを利用できません')
        if mode == 'linear':
            budget = 100
        with self.lock:
            # Bounded local sessions; evict only inactive old sessions.
            for key, g in list(self.games.items()):
                if time.monotonic()-g['touched'] > 3600 and not g['busy']:
                    del self.games[key]
            if len(self.games) >= 32:
                raise ValueError('対局数が上限です。サーバーを再起動してください')
            key = secrets.token_urlsafe(24)
            self.games[key] = dict(state=State(), seat=seat, mode=mode, budget=budget, history=[], busy=False, error=None, stats={}, touched=time.monotonic(), rng=random.Random(1))
            self.advance(key)
            return self.snapshot(key)

    def advance(self, key):
        g = self.games[key]
        s = g['state']
        # Automatic forced passes, including the human seat.
        while not s.done and not legal_moves(s):
            play(s, None, validate=False)
            g['history'].append(None)
        if not s.done and s.turn != g['seat'] and not g['busy']:
            g['busy'] = True
            self.worker.submit(self.think, key, copy.deepcopy(s))

    def think(self, key, state):
        g = None
        try:
            with self.lock:
                g = self.games[key]
                revision = len(g['history'])
            agent = self.linear_factory() if g['mode']=='linear' else SearchAgent(self.bot, g['budget']) if g['mode']=='search' else self.bot
            start = time.perf_counter()
            move = agent.act(state, g['rng'])
            elapsed = (time.perf_counter()-start)*1000
            if g['mode'] == 'linear':
                detail = agent.stats[-1]
                stats = dict(elapsed_ms=elapsed, depth=detail.get('depth', 0), fallback=detail.get('fallback', False), target_complete=detail.get('target_complete', True))
            else:
                stats = agent.stats if g['mode']=='search' else dict(elapsed_ms=elapsed)
            with self.lock:
                if self.games.get(key) is not g or len(g['history']) != revision or g['state'] != state:
                    return  # Never apply a result to a replaced/advanced session.
                play(g['state'], move, validate=True)
                g['history'].append(None if move is None else [move.piece, move.orientation, move.row, move.col])
                g['stats'] = stats
                g['busy'] = False
                self.advance(key)
        except Exception as error:
            with self.lock:
                if g is not None and self.games.get(key) is g:
                    g['busy'] = False
                    g['error'] = str(error)

    def move(self, key, revision, values):
        with self.lock:
            g = self.games[key]
            if g['busy'] or g['error'] or g['state'].turn != g['seat'] or revision != len(g['history']):
                raise ValueError('盤面が更新されています。表示を更新してください')
            if not isinstance(values, list) or len(values)!=4 or any(type(v) is not int for v in values):
                raise ValueError('配置が不正です')
            move = Move(*values)
            play(g['state'], move, validate=True)
            g['history'].append(values)
            self.advance(key)
            return self.snapshot(key)

    def snapshot(self, key):
        with self.lock:
            g = self.games[key]
            g['touched'] = time.monotonic()
            s = g['state']
            qs = legal_moves(s) if s.turn==g['seat'] and not g['busy'] else []
            return dict(id=key, revision=len(g['history']), boards=[list(cells_of(b)) for b in s.boards], remaining=[[i for i in range(21) if s.remaining[p]>>i&1] for p in (0,1)], scores=[s.score(p) for p in (0,1)], turn=s.turn, seat=g['seat'], done=s.done, busy=g['busy'], error=g['error'], mode=g['mode'], budget=g['budget'], generation='linear-L' if g['mode']=='linear' else 'gen3', model_seed=3101 if g['mode']=='linear' else 1751401, stats=g['stats'], legal=[[q.move.piece,q.move.orientation,q.move.row,q.move.col] for q in qs], history=g['history'])


def handler(games):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == '/':
                data = Path(__file__).with_name('web_play.html').read_bytes()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif self.path == '/api/pieces':
                self.reply(200, dict(pieces=ORIENTS, starts=START))
            elif self.path.startswith('/api/game/'):
                try:
                    self.reply(200, games.snapshot(self.path.split('/')[-1]))
                except KeyError:
                    self.reply(404, dict(error='対局が見つかりません'))
            else:
                self.reply(404, dict(error='not found'))

        def do_POST(self):
            # JSON-only requests with same-origin checks; no CORS.
            origin = self.headers.get('Origin')
            if origin and origin != 'http://' + self.headers.get('Host', ''):
                self.reply(403, dict(error='origin mismatch')); return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 8192 or self.headers.get('Content-Type','').split(';')[0]!='application/json':
                    raise ValueError('invalid request')
                body = json.loads(self.rfile.read(length))
                if self.path == '/api/new':
                    result = games.create(body['seat'], body['mode'], body['budget'])
                elif self.path == '/api/move':
                    result = games.move(body['id'], body['revision'], body['move'])
                else:
                    self.reply(404, dict(error='not found')); return
                self.reply(200, result)
            except (ValueError, KeyError, TypeError) as error:
                self.reply(400, dict(error=str(error)))
    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--device', choices=['cpu','cuda','auto'], default='cpu')
    args = parser.parse_args()
    import torch
    from .ppo_policy import PPOAgent
    torch.set_num_threads(1)
    from .linear_inference import load_search, make_search
    load_start = time.perf_counter()
    bot = PPOAgent(device=args.device, backend='auto')
    linear = load_search()
    print(f'Model load: {(time.perf_counter()-load_start)*1000:.1f} ms', flush=True)
    # Load/feature initialization is done before accepting games; log cold cost.
    for name, agent in [('gen3', bot), ('linear-L', linear)]:
        begin = time.perf_counter()
        agent.act(State(), random.Random(0))
        if name == 'linear-L':
            agent.agent.value(State())
        print(f'Warmup {name}: {(time.perf_counter()-begin)*1000:.1f} ms', flush=True)
    games = Games(bot, lambda: make_search(linear.agent))
    server = ThreadingHTTPServer((args.host, args.port), handler(games))
    print(f'Blokus: http://{args.host}:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        games.worker.shutdown(wait=True)

if __name__ == '__main__':
    main()
