"""Read-only browser interactions against a running console."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

out = Path('artifacts/console-review')
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path='/usr/bin/google-chrome', headless=True, args=['--no-sandbox'])
    page = browser.new_page(viewport={'width':1440,'height':1100})
    errors=[]
    page.on('pageerror', lambda e: errors.append(str(e)))
    page.goto('http://100.95.30.96:8766/')
    page.wait_for_function('replay !== null && catalog !== undefined')
    assert page.locator('#group option').count()==44
    page.click('#step'); assert page.locator('#seek').input_value()=='1'
    page.click('#back'); assert page.locator('#seek').input_value()=='0'
    page.select_option('#speed','200')
    page.click('#play'); page.wait_for_function('ply>=3')
    page.click('#play'); before=page.locator('#seek').input_value()
    page.wait_for_timeout(500); assert page.locator('#seek').input_value()==before
    page.locator('#seek').evaluate('(el)=>{el.value=el.max;el.dispatchEvent(new Event("input"))}')
    assert '対局終了' in page.locator('#state').inner_text()
    assert '勝ち' in page.locator('#state').inner_text() or '引き分け' in page.locator('#state').inner_text()
    # Start playback from penultimate position, then verify automatic next game.
    page.click('#play')
    page.locator('#seek').evaluate('(el)=>{el.value=Number(el.max)-1;el.dispatchEvent(new Event("input"))}')
    page.click('#play'); page.wait_for_function('document.getElementById("game").value==="1" && replay !== null')
    page.click('#play')
    page.select_option('#phase','validation')
    page.select_option('#opponent','gen3 / Frozen')
    assert page.locator('#evaluations tr').count()==3
    page.select_option('#phase','confirmation')
    page.select_option('#opponent','配置価値A / 24,576局')
    assert page.locator('#evaluations tr').count()==1
    page.select_option('#opponent','gen3 / Frozen')
    page.click('#step')
    page.screenshot(path=str(out/'desktop.png'),full_page=True)
    archives=page.locator('#group option').evaluate_all('(xs)=>xs.filter(o=>o.value.startsWith("archive:")).map(o=>o.value)')
    page.select_option('#group',archives[0]);page.wait_for_function('replay!==null && replay.kind==="人間対AI"')
    assert '人間' in page.locator('#match').inner_text()
    page.select_option('#group',page.locator('#group option').first.get_attribute('value'))
    page.wait_for_function('replay!==null && replay.kind==="AI評価対戦"')
    page.set_viewport_size({'width':390,'height':844})
    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
    page.screenshot(path=str(out/'mobile.png'),full_page=True)
    assert not errors, errors
    assert not page.locator('#error').inner_text()
    # Server rejects mutation and arbitrary source file access.
    assert page.request.post('http://100.95.30.96:8766/api/new',data={}).status==501
    assert page.request.get('http://100.95.30.96:8766/api/replay?group=../../etc/passwd').status==404
    print(json.dumps(dict(browser_errors=errors, desktop=[1440,1100], mobile=[390,844],
                         checks=['step','back','speed','play','pause','seek','result','continuous next game',
                                 'phase/opponent chart filters','human archive','responsive layout','no mutation API']),indent=2))
    browser.close()
