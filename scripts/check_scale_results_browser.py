"""Final scale results and real recorded predictions, including terminal hold."""
import json,time
from pathlib import Path
from playwright.sync_api import sync_playwright
R=Path('artifacts/komi4-scale')
with sync_playwright() as p:
    b=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox'])
    page=b.new_page(viewport={'width':1440,'height':1000});errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto('http://100.95.30.96:8766/');page.wait_for_function('scale?.results && replay')
    assert page.evaluate('scale.results.unique_train_games')==11000
    assert page.locator('#scale-matches tr').count()==6
    assert page.locator('#scale-curve polyline').count()==3
    assert '独立群不足' in page.locator('#scale-matches').inner_text()
    assert 'ridge' in page.locator('#scale-prediction').inner_text()
    page.locator('#campaign').screenshot(path=str(R/'final-desktop.png'))
    page.click('#scale-watch');page.wait_for_function('replay?.source.includes("comparison")')
    def seek(expr):page.locator('#seek').evaluate('(el,e)=>{el.value=eval(e);el.dispatchEvent(new Event("input"));}',expr)
    seek('replay.prefix_plies')
    assert '当該対局で保存した途中予測' in page.locator('#point-prediction').inner_text()
    page.select_option('#perspective','0');first=page.locator('#point-prediction').inner_text()
    page.select_option('#perspective','1');second=page.locator('#point-prediction').inner_text()
    assert '青・先手' in first and '黄・後手' in second
    import re
    a=float(re.search(r'途中予測 ([\-\d.]+)点',first)[1]);c=float(re.search(r'途中予測 ([\-\d.]+)点',second)[1]);assert abs(a+c)<.001
    seek('0');assert '未記録' in page.locator('#point-prediction').inner_text()
    page.select_option('#speed','200');seek('replay.moves.length-1');page.click('#play');page.wait_for_function('ply===replay.moves.length');start=time.monotonic()
    old=page.locator('#game').input_value();assert '後手+4点適用' in page.locator('#result').inner_text()
    page.select_option('#speed','2000');page.wait_for_timeout(2500);assert page.locator('#game').input_value()==old
    page.wait_for_function('(old)=>document.getElementById("game").value!==old && replay!==null',arg=old)
    elapsed=time.monotonic()-start;assert 2.85<=elapsed<4.2
    page.click('#play');page.set_viewport_size({'width':390,'height':844});assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
    page.locator('#campaign').screenshot(path=str(R/'final-mobile.png'))
    assert not errors
    result=dict(checks=['11000 unique train games','6 final comparison rows','3 learning curves','recorded prediction perspective inversion','prefix missing','komi4 final score','terminal hold independent of speed','mobile width'],terminal_seconds=elapsed,perspective_values=[a,c],errors=errors)
    (R/'browser-final.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result));b.close()
