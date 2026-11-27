"""Check recorded perspectives and terminal hold using real replay and wall time."""
import json,time
from pathlib import Path
from playwright.sync_api import sync_playwright
out=Path('artifacts/console-review')
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1440,'height':1100});errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto('http://100.95.30.96:8766/')
    page.wait_for_function('replay && replay.evaluations!==undefined')
    def seek(expr):
        page.locator('#seek').evaluate('(el,expression)=>{el.value=eval(expression);el.dispatchEvent(new Event("input"));}',expr)
    seek('replay.prefix_plies')
    original=page.evaluate('replay.evaluations[ply]')
    assert original['value'] is not None
    page.select_option('#perspective',str(original['player']))
    a=float(page.locator('#evaluation-value').inner_text())
    page.select_option('#perspective',str(1-original['player']))
    b=float(page.locator('#evaluation-value').inner_text());assert abs(a+b)<.00001
    assert '符号反転' in page.locator('#evaluation-detail').inner_text()
    page.click('#step');next_eval=page.evaluate('replay.evaluations[ply]')
    assert next_eval['player']!=original['player']
    page.select_option('#perspective','mover')
    assert abs(float(page.locator('#evaluation-value').inner_text())-next_eval['value'])<.0001
    page.screenshot(path=str(out/'evaluation-desktop.png'),full_page=True)
    seek('0');assert page.locator('#evaluation-value').inner_text()=='未記録'
    page.select_option('#speed','200')
    seek('replay.moves.length-1');page.click('#play')
    page.wait_for_function('ply===replay.moves.length');start=time.monotonic()
    old=page.locator('#game').input_value()
    assert page.locator('#result').is_visible()
    assert '点差' in page.locator('#result').inner_text()
    result=page.locator('#result').inner_text()
    page.screenshot(path=str(out/'terminal-result.png'),full_page=True)
    # Change speed while holding: original end deadline must be preserved.
    page.select_option('#speed','2000')
    time.sleep(max(0,2.6-(time.monotonic()-start)))
    assert page.locator('#game').input_value()==old
    page.wait_for_function('(old)=>document.getElementById("game").value!==old && replay!==null',arg=old)
    elapsed=time.monotonic()-start;assert 2.85<=elapsed<4.2,elapsed
    page.click('#play')
    # Pausing during terminal hold prevents the queued transition.
    page.select_option('#speed','200');seek('replay.moves.length-1');page.click('#play')
    page.wait_for_function('ply===replay.moves.length');page.click('#play');old=page.locator('#game').input_value()
    page.wait_for_timeout(3200);assert page.locator('#game').input_value()==old
    # Resuming at terminal holds again; disabling continuous cancels that wait.
    page.click('#play');page.locator('#continuous').uncheck()
    page.wait_for_timeout(3200);assert page.locator('#game').input_value()==old
    assert page.locator('#play').inner_text()=='再生'
    # Manual step to terminal never schedules next game, even with continuous on.
    page.locator('#continuous').check();seek('replay.moves.length-1');page.click('#step')
    page.wait_for_timeout(3200);assert page.locator('#game').input_value()==old
    assert page.locator('#result').is_visible()
    # Continuous off during ordinary playback leaves the terminal result visible.
    page.locator('#continuous').uncheck();seek('replay.moves.length-1');page.click('#play')
    page.wait_for_function('ply===replay.moves.length && !playing')
    page.wait_for_timeout(3200);assert page.locator('#game').input_value()==old
    archive=page.locator('#group option').evaluate_all('(xs)=>xs.find(o=>o.value.startsWith("archive:") && o.textContent.includes("gen4")).value')
    page.select_option('#group',archive);page.wait_for_function('replay!==null && replay.kind==="人間対AI"')
    seek('1');assert page.locator('#evaluation-value').inner_text()=='未記録'
    assert '残っていません' in page.locator('#evaluation-detail').inner_text()
    page.set_viewport_size({'width':390,'height':844})
    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
    page.screenshot(path=str(out/'evaluation-mobile.png'),full_page=True)
    assert not errors,errors
    print(json.dumps(dict(terminal_hold_seconds=elapsed,terminal_text=result,perspective_values=[a,b],
          checks=['mover/first/second perspective','actual next mover','prefix missing','human archive missing',
          'terminal 3 seconds independent of speed','pause cancels transition','continuous off cancels transition',
          'manual terminal stays','continuous off during playback','mobile width'],browser_errors=errors),ensure_ascii=False,indent=2))
    browser.close()
