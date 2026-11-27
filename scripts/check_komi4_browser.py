"""Live VPN UI checks with real saved research games and terminal wall time."""
import json,time
from pathlib import Path
from playwright.sync_api import sync_playwright
R=Path('artifacts/komi4-campaign')
with sync_playwright() as p:
 b=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox']);page=b.new_page(viewport={'width':1440,'height':1100});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
 page.goto('http://100.95.30.96:8766/');page.wait_for_function('replay && campaign?.plan && catalog.groups.some(g=>g.id==="campaign:train")')
 page.select_option('#group','campaign:train');page.wait_for_function('replay?.komi===4')
 def seek(expr):page.locator('#seek').evaluate('(el,e)=>{el.value=eval(e);el.dispatchEvent(new Event("input"));}',expr)
 seek('0');assert '未記録' in page.locator('#point-prediction').inner_text()
 seek('replay.prefix_plies');page.select_option('#perspective','0')
 a=page.evaluate('replay.predictions.values[ply]');assert '後手+4点' in page.locator('#point-prediction').inner_text();assert page.locator('#prediction-komi').is_disabled()
 original=page.locator('#point-prediction').inner_text();page.select_option('#perspective','1');assert '黄・後手視点' in page.locator('#point-prediction').inner_text();assert original!=page.locator('#point-prediction').inner_text()
 expected=a['adjusted']*(1 if a['player']==1 else -1);assert f'{expected:.2f}点' in page.locator('#point-prediction').inner_text()
 if page.evaluate('replay.evaluations[ply].value!==null'):
  x=float(page.locator('#evaluation-value').inner_text().replace(' 点',''));page.select_option('#perspective','0');y=float(page.locator('#evaluation-value').inner_text().replace(' 点',''));assert abs(x+y)<.001
 seek('replay.moves.length');assert '後手+4点適用' in page.locator('#result').inner_text();assert '終局の正確値' in page.locator('#point-prediction').inner_text()
 result=page.locator('#result').inner_text();scores=page.evaluate('replay.frames[ply].scores');assert f'{abs(scores[0]-scores[1]-4)}点差' in result
 page.screenshot(path=str(R/'terminal-desktop.png'),full_page=True)
 page.select_option('#speed','200');seek('replay.moves.length-1');page.click('#play');page.wait_for_function('ply===replay.moves.length');start=time.monotonic();old=page.locator('#game').input_value()
 page.select_option('#speed','2000');page.wait_for_timeout(2500);assert page.locator('#game').input_value()==old
 page.wait_for_function('(old)=>document.getElementById("game").value!==old && replay!==null',arg=old);elapsed=time.monotonic()-start;assert 2.85<elapsed<4.2
 page.click('#play');page.select_option('#speed','200');seek('replay.moves.length-1');page.click('#play');page.wait_for_function('ply===replay.moves.length');page.click('#play');old=page.locator('#game').input_value();page.wait_for_timeout(3100);assert page.locator('#game').input_value()==old
 page.click('#play');page.locator('#continuous').uncheck();page.wait_for_timeout(3100);assert page.locator('#game').input_value()==old
 page.locator('#continuous').check();seek('replay.moves.length-1');page.click('#step');page.wait_for_timeout(3100);assert page.locator('#game').input_value()==old
 page.locator('#campaign').scroll_into_view_if_needed();page.screenshot(path=str(R/'campaign-desktop.png'),full_page=True)
 page.set_viewport_size({'width':390,'height':844});assert page.evaluate('document.documentElement.scrollWidth<=innerWidth');page.screenshot(path=str(R/'campaign-mobile.png'),full_page=True)
 # Old saved evaluations retain komi0 behavior and editable display-only correction.
 page.select_option('#group','experiment:paired-00-0');page.wait_for_function('replay?.source.includes("paired-00-0")');seek('replay.prefix_plies');assert not page.locator('#prediction-komi').is_disabled()
 assert not errors,errors
 write=dict(terminal_hold_seconds=elapsed,terminal_text=result,checks=['saved prediction perspective sign','recorded search perspective sign','prefix missing','fixed research komi4','bonus-aware terminal margin','terminal three seconds with speed change','pause cancels','continuous off cancels','manual step stays','old komi0 independent','mobile width'],browser_errors=errors)
 print(json.dumps(write,ensure_ascii=False,indent=2));b.close()
