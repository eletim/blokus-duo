"""Read-only observation server: no inference, training or archive writes."""
import argparse
from datetime import datetime, timezone
from functools import lru_cache
import gzip
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
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


def recorded_evaluation(stats, player, model, move, *, trusted=False, reason=None):
    """DeepSearch chooses argmax(root_scores) of the last completed iteration.

    The root scores already use the mover's perspective. Do not select by
    piece id, average roots, or regenerate the candidate order with inference.
    Only the indexed evaluation shards have this audited producer contract.
    """
    stats = stats if isinstance(stats, dict) else {}
    result = dict(value=None, player=player, model=model, move=move,
                  depth=stats.get('depth'), reason=reason or '選択手の評価値は未記録',
                  meaning='勝敗符号を教師にした探索評価（勝率・予測点差ではない）')
    values = stats.get('root_scores')
    if not trusted:
        return result
    if stats.get('player') != player:
        result['reason'] = '保存statsの手番が棋譜と不一致'
        return result
    if move is None or stats.get('forced_pass'):
        result['reason'] = '強制パスの探索評価は未記録'
    elif stats.get('fallback') is not False:
        result['reason'] = '探索未完了・代替手の評価は未記録'
    elif (not isinstance(values, list) or not values or
          not all(type(v) in (float, int) and math.isfinite(v) for v in values) or
          stats.get('depth') not in (2, 4, 6) or
          not stats.get('stages') or stats['stages'][-1] != stats['depth']):
        result['reason'] = '完了した探索段階の評価値は未記録'
    else:
        selected = max(range(len(values)), key=values.__getitem__)
        result.update(value=values[selected], root_index=selected,
                      candidate_count=len(values), reason=None,
                      extraction='最後の完了深さのroot_scores最大値（同点は最初）')
    return result


RESEARCH = Path(__file__).resolve().parents[1]/'artifacts/margin-pilot'
CAMPAIGN = RESEARCH.parent/'komi4-campaign'


def point_predictions(moves, scores, source='', index=0):
    model = RESEARCH/'model.json'
    if not model.exists(): return None
    from .margin_value import MarginValue
    evaluator=MarginValue(model);s=State();out=[]
    for m in moves+[None]:
        out.append(dict(player=s.turn, raw=evaluator.raw(s), actual=scores[s.turn]-scores[1-s.turn],
                        exact=s.done))
        if len(out)>len(moves):break
        play(s,Move(*m) if m is not None else None,validate=False)
    mapping=RESEARCH/'prediction-splits.json'
    split=read_json(mapping).get(f'{source}#{index}', '追加実験・学習外') if mapping.exists() else '未取得'
    return dict(values=out, split=split, model='margin-ridge-v1',
                meaning='固定した実験モデルによる後付け推論。過去AIの当時評価ではない。')


def research_data():
    result={}
    for key,name in [('existing','existing-analysis.json'),('training','training-result.json'),('matches','match-analysis.json')]:
        path=RESEARCH/name
        result[key]=read_json(path) if path.exists() else None
    return result


def campaign_data():
    result={}
    for name in ('plan','status','selection','results','audit'):
        path=CAMPAIGN/(name+'.json')
        result[name]=read_json(path) if path.exists() else None
    result['games']={split:len(campaign_games(split)) for split in ('train','validation','test','comparison')}
    result['training']=[read_json(p) for p in sorted(CAMPAIGN.glob('gen*-result.json'))]
    result['updated']=datetime.now(timezone.utc).isoformat()
    return result


def campaign_games(split):
    if split not in ('train','validation','test','comparison'):return []
    return [p for p in sorted((CAMPAIGN/'games').glob('*.json')) if read_json(p)['task']['split']==split]


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
        for path in sorted((RESEARCH/'matches').glob('*.json')):
            d=read_json(path);t=d['task']
            groups.append(dict(id='experiment:'+path.stem,stage=0,phase='experiment',seed=t['seed'],count=1,
                left='gen4同士・初期盤面' if t['kind']=='self' else '点差実験版 vs gen4',
                right=f"先後条件 {t['seat']} / {path.stem}"))
        for split in ('train','validation','test','comparison'):
            files=campaign_games(split)
            if files:groups.append(dict(id='campaign:'+split,stage=0,phase='komi4',seed='3 seeds',count=len(files),
                left='非線形点差 / コミ4',right={'train':'追加学習用','validation':'選択用','test':'未知予測評価用','comparison':'最終比較'}[split]))
        return dict(groups=groups, evaluations=evaluations, training=training, progress=progress,
                    archive=archived, errors=errors, pieces=PIECES,
                    updated=datetime.now(timezone.utc).isoformat())

    def game(self, group, index):
        if group.startswith('campaign:'):
            return self.campaign(group.removeprefix('campaign:'),index)
        if group.startswith('experiment:'):
            return self.experiment(group.removeprefix('experiment:'))
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
        snapshots = frames(moves, game['scores'])
        evaluations = []
        stats = game.get('stats', [])
        aligned = len(stats) == len(game['moves'])
        for ply, move in enumerate(moves):
            player = snapshots[ply]['turn']
            recorded = stats[ply-len(prefix)] if aligned and ply >= len(prefix) else None
            evaluations.append(recorded_evaluation(recorded, player, players[player], move,
                trusted=aligned and ply >= len(prefix),
                reason='共通開始局面の評価は未記録' if ply < len(prefix) else None))
        evaluations.append(dict(value=None, reason='終局：探索評価は未記録（最終結果は実測）'))
        return dict(players=players, seed=data['seed'], prefix_plies=len(prefix), moves=moves,
                    evaluations=evaluations, predictions=point_predictions(moves,game['scores'],group,index),
                    frames=snapshots, source=group, index=index,
                    count=len(data['games']), kind='AI評価対戦', budget_ms=data['budget_ms'])

    def human_game(self, key):
        if not self.archive.exists(): raise KeyError('棋譜は未取得です')
        with sqlite3.connect(self.archive.as_uri()+'?mode=ro', uri=True) as db:
            row = db.execute("SELECT record FROM games WHERE id=? AND kind='human'", (key,)).fetchone()
        if row is None: raise KeyError('棋譜が見つかりません')
        r = json.loads(row[0]); players = [r['generation'], r['generation']]
        players[r['human_seat']] = '人間'
        moves = [m['move'] for m in r['moves']]
        snapshots = frames(moves, r['scores'])
        evaluations = [recorded_evaluation(item.get('stats'), snapshots[i]['turn'],
                        players[snapshots[i]['turn']], item['move'],
                        reason='人間の手：AI評価は未記録' if item.get('actor') == 'human'
                        else 'Web保存棋譜には選択手の評価値が残っていません')
                       for i, item in enumerate(r['moves'])]
        evaluations.append(dict(value=None, reason='終局または保存末尾：探索評価は未記録'))
        return dict(players=players, seed=r.get('model_seed'), prefix_plies=0, moves=moves,
                    evaluations=evaluations, predictions=point_predictions(moves,r['scores']) if r['status']=='finished' else None,
                    frames=snapshots, source='人間対AI 保存棋譜', index=0,
                    count=1, kind='人間対AI', budget_ms=r.get('budget_ms'))


    def experiment(self,key):
        path=next((p for p in (RESEARCH/'matches').glob('*.json') if p.stem==key),None)
        if path is None:raise KeyError('実験棋譜がありません')
        d=read_json(path);t=d['task'];moves=d['moves'];snapshots=frames(moves,d['scores'])
        players=['gen4 / Frozen','gen4 / Frozen']
        if t['kind']=='paired':players[t['seat']]='点差実験版 / margin-ridge-v1'
        evaluations=[];prefix=len(t['prefix'])
        for i,m in enumerate(moves):
            p=snapshots[i]['turn'];st=d['stats'][i-prefix] if i>=prefix else None
            e=recorded_evaluation(st,p,players[p],m,trusted=i>=prefix,
                                 reason='共通開始局面の評価は未記録' if i<prefix else None)
            if st and st['agent']=='margin':e['unit']='points'
            evaluations.append(e)
        evaluations.append(dict(value=None,reason='終局：最終スコアは実測'))
        return dict(players=players,seed=t['seed'],prefix_plies=prefix,moves=moves,frames=snapshots,
                    evaluations=evaluations,predictions=point_predictions(moves,d['scores']),
                    source='追加の独立実験: '+key,index=0,count=1,kind='実験対戦',budget_ms=500)

    def campaign(self,split,index):
        files=campaign_games(split)
        if not 0<=index<len(files):raise KeyError('研究対局がありません')
        path=files[index];d=read_json(path);t=d['task'];moves=d['moves'];snapshots=frames(moves,d['scores']);prefix=len(t['prefix'])
        generation=1 if split=='comparison' else 0
        players=['',''];players[t['seat']]=f'非線形点差 / 世代{generation}'
        players[1-t['seat']]={'self':f'非線形点差 / 世代{generation}','ridge':'旧ridge点差','gen4':'gen4 / 凍結ヒューリスティック'}[t['opponent']]
        evaluations=[];predictions=[None]*(len(moves)+1)
        for i,m in enumerate(moves):
            p=snapshots[i]['turn'];st=d['stats'][i-prefix] if i>=prefix else None
            e=recorded_evaluation(st,p,players[p],m,trusted=i>=prefix,reason='ランダム開始手順の評価は未記録' if i<prefix else None)
            if st and st['agent']!='gen4':e['unit']='points'
            e['komi']=4;evaluations.append(e)
        evaluations.append(dict(value=None,reason='終局：コミ4での正確な実測点差'))
        for entry in d['predictions']:
            p=entry['player'];predictions[entry['ply']]=dict(player=p,adjusted=entry['adjusted'],actual=d['adjusted_first_margin']*(1 if p==0 else -1),exact=False)
        p=snapshots[-1]['turn'];actual=d['adjusted_first_margin']*(1 if p==0 else -1)
        predictions[-1]=dict(player=p,adjusted=actual,actual=actual,exact=True)
        return dict(players=players,seed=t['seed'],prefix_plies=prefix,moves=moves,frames=snapshots,evaluations=evaluations,
            predictions=dict(values=predictions,model=f'非線形点差 / 世代{generation} / seed {t["seed"]}',split=split,komi=4,recorded=True),
            komi=4,source='コミ4研究 / '+path.stem,index=index,count=len(files),kind='独立研究・後手+4点',budget_ms=500 if split=='comparison' else 100)


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
                elif url.path == '/api/campaign': self.respond(200,campaign_data())
                elif url.path == '/api/research': self.respond(200,research_data())
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
