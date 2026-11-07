"""Transactional, local-only server snapshots and operator CLI (no Torch required)."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
from .engine import State, Move, play, START


def now():
    return datetime.now(timezone.utc).isoformat()


def identity():
    root = Path(__file__).resolve().parents[1]
    try:
        version = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        version = 'unknown'
    paths = {'gen3': root/'artifacts/ppo-gen3/seed-1751401.pt',
             'linear-L': root/'models/linear-L/seed-3101.json'}
    return dict(code_version=version, model_hashes={k: hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()})


class Archive:
    def __init__(self, data_dir):
        self.path = Path(data_dir).expanduser()/'games.sqlite3'

    @contextmanager
    def connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path, timeout=3)
        try:
            with db:
                db.execute('PRAGMA synchronous=FULL')
                db.execute('CREATE TABLE IF NOT EXISTS games (id TEXT PRIMARY KEY, started TEXT NOT NULL, updated TEXT NOT NULL, mode TEXT NOT NULL, kind TEXT NOT NULL, revision INTEGER NOT NULL, record TEXT NOT NULL)')
                yield db
        finally:
            db.close()

    def save(self, record):
        # Snapshot replaces only older revisions, in one committed transaction.
        with self.connect() as db:
            db.execute('INSERT INTO games VALUES (?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET updated=excluded.updated, revision=excluded.revision, record=excluded.record WHERE excluded.revision>=games.revision',
                       (record['id'],record['started_at'],record['updated_at'],record['mode'],record['kind'],record['revision'],json.dumps(record,ensure_ascii=False)))

    def get(self, key):
        with self.connect() as db:
            row=db.execute('SELECT record FROM games WHERE id=?',(key,)).fetchone()
        if row is None: raise KeyError(key)
        return json.loads(row[0])

    def listing(self, mode=None, since=None, kind=None):
        with self.connect() as db:
            return [dict(zip(('id','started_at','updated_at','mode','kind','revision'),r)) for r in db.execute('SELECT id,started,updated,mode,kind,revision FROM games WHERE (? IS NULL OR mode=?) AND (? IS NULL OR started>=?) AND (? IS NULL OR kind=?) ORDER BY started', (mode,mode,since,since,kind,kind))]


def record(key,g,meta):
    s=g['state']; generation='linear-L' if g['mode']=='linear' else 'gen3'
    scores=[s.score(0),s.score(1)]
    moves=[dict(ply=i+1,turn=i%2,move=m,actor='human' if i%2==g['seat'] else 'ai',stats=g['move_stats'][i]) for i,m in enumerate(g['history'])]
    return dict(schema_version=1,id=key,started_at=g['started_at'],updated_at=now(),kind=g['kind'],rules=dict(name='Blokus Duo',board=[14,14],starts=START,coordinates='zero-based',all_placed_bonus=15,last_monomino_bonus=20),human_seat=g['seat'],mode=g['mode'],model_seed=3101 if generation=='linear-L' else 1751401,generation=generation,model_sha256=meta.get('model_hashes',{}).get(generation),code_version=meta.get('code_version','unknown'),budget_ms=g['budget'],revision=len(moves),moves=moves,status='finished' if s.done else 'unfinished',scores=scores,winner=(None if scores[0]==scores[1] else int(scores[1]>scores[0])) if s.done else None,result=('draw' if scores[0]==scores[1] else 'human_win' if scores[g['seat']]>scores[1-g['seat']] else 'ai_win') if s.done else None,state=dict(boards=s.boards,remaining=s.remaining,last=s.last,turn=s.turn,passes=s.passes))


def replay(r):
    s=State()
    for i,item in enumerate(r['moves']):
        assert item['ply']==i+1 and item['turn']==s.turn
        m=item['move'];play(s,Move(*m) if m is not None else None,validate=True)
    assert r['revision']==len(r['moves'])
    assert r['scores']==[s.score(0),s.score(1)]
    assert (r['status']=='finished')==s.done
    assert r['state']==dict(boards=s.boards,remaining=s.remaining,last=s.last,turn=s.turn,passes=s.passes)
    if s.done:
        scores=r['scores'];assert r['winner']==(None if scores[0]==scores[1] else int(scores[1]>scores[0]))
    return dict(id=r['id'],valid=True,status=r['status'],plies=len(r['moves']),scores=r['scores'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-dir',default=os.environ.get('BLOKUS_DATA_DIR',str(Path.home()/'.local/share/blokus-web')))
    p.add_argument('command',choices=['list','export','replay']);p.add_argument('--id');p.add_argument('--mode');p.add_argument('--since');p.add_argument('--kind',choices=['human','test']);p.add_argument('--output');a=p.parse_args();db=Archive(a.data_dir)
    if a.command=='list':out=db.listing(a.mode,a.since,a.kind)
    else:
        if not a.id:p.error('--id is required')
        r=db.get(a.id);out=replay(r) if a.command=='replay' else r
    text=json.dumps(out,ensure_ascii=False,indent=2)
    if a.output:Path(a.output).write_text(text+'\n')
    else:print(text)

if __name__=='__main__':main()
