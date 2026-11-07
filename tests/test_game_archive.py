import copy,json,subprocess,sys,time,threading
import pytest
from blokus.game_archive import Archive,replay
from blokus.web_play import Games
from test_web_play import First
from test_web_linear import wait


def test_full_games_sessions_restart_and_export(tmp_path):
 db=Archive(tmp_path);games=Games(First(),archive=db,record_kind='test')
 try:
  a=games.create(0,'policy',500);b=games.create(1,'policy',500)
  assert db.get(a['id'])['revision']==0 and db.get(a['id'])['status']=='unfinished'
  for key in (a['id'],b['id']):
   while True:
    g=wait(games,key)
    if g['done']:break
    prev=g['revision'];move=g['legal'][0];games.move(key,prev,move)
    with pytest.raises(ValueError):games.move(key,prev,move)
   r=db.get(key);assert replay(r)['valid'] and r['kind']=='test'
   assert any(x['move'] is None for x in r['moves'])
   assert r['result'] in ('draw','human_win','ai_win')
   out=subprocess.check_output([sys.executable,'-m','blokus.game_archive','--data-dir',str(tmp_path),'export','--id',key],text=True)
   assert json.loads(out)==r
   subprocess.check_call([sys.executable,'-m','blokus.game_archive','--data-dir',str(tmp_path),'replay','--id',key])
  assert len(Archive(tmp_path).listing(kind='test',mode='policy'))==2
  assert Archive(tmp_path).listing(kind='human')==[]
 finally:games.worker.shutdown()


def test_failure_is_visible_and_full_snapshot_recovers(tmp_path):
 db=Archive(tmp_path);original=db.save
 def fail(r):raise OSError('disk full')
 db.save=fail;games=Games(First(),archive=db)
 try:
  g=games.create(0,'policy',500);assert g['archive']['failed'] and g['archive']['saved_revision'] is None
  db.save=original
  games.move(g['id'],0,g['legal'][0]);g=wait(games,g['id'])
  assert not g['archive']['failed'] and g['archive']['saved_revision']==g['revision']
  assert replay(db.get(g['id']))['valid']
 finally:games.worker.shutdown()


def test_stale_worker_never_written_and_unfinished_survives(tmp_path):
 entered=threading.Event();release=threading.Event()
 class Slow(First):
  def act(self,s,rng):entered.set();assert release.wait(5);return super().act(s,rng)
 db=Archive(tmp_path);games=Games(Slow(),archive=db)
 try:
  a=games.create(1,'policy',500);assert entered.wait(5)
  with games.lock:
   replacement=copy.deepcopy(games.games[a['id']]);replacement['busy']=False
   games.games[a['id']]=replacement
  release.set();games.worker.shutdown()
  r=db.get(a['id']);assert r['revision']==0 and r['status']=='unfinished' and r['result'] is None
  assert replay(r)['valid']
 finally:release.set();games.worker.shutdown()


def test_transaction_preserves_old_record_on_error(tmp_path):
 db=Archive(tmp_path);g=Games(First(),archive=db)
 try:
  a=g.create(0,'policy',500);old=db.get(a['id'])
  with pytest.raises(RuntimeError):
   with db.connect() as con:
    con.execute('DELETE FROM games');raise RuntimeError('rollback')
  assert db.get(a['id'])==old
 finally:g.worker.shutdown()
