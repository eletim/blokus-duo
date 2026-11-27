import json
from pathlib import Path
from playwright.sync_api import sync_playwright
R=Path('artifacts/komi4-scale')
with sync_playwright() as p:
 b=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox']);page=b.new_page(viewport={'width':1440,'height':1000});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
 page.goto('http://100.95.30.96:8766/');page.wait_for_function('scale?.plan && replay')
 assert page.evaluate('scale.plan.minimum_unique_train_total')==10000
 assert 'validation' in page.locator('#scale-counts').inner_text()
 assert '実学習' in page.locator('#scale-time').inner_text() and '総経過' in page.locator('#scale-time').inner_text()
 page.click('#scale-watch');page.wait_for_function('replay?.source.startsWith("拡大学習")')
 page.locator('#seek').evaluate('(el)=>{el.value=replay.prefix_plies;el.dispatchEvent(new Event("input"));}')
 assert '未記録' in page.locator('#point-prediction').inner_text()
 assert page.evaluate('replay.komi')==4
 page.locator('#seek').evaluate('(el)=>{el.value=el.max;el.dispatchEvent(new Event("input"));}')
 assert '後手+4点適用' in page.locator('#result').inner_text()
 page.locator('#campaign').screenshot(path=str(R/'progress-desktop.png'))
 page.set_viewport_size({'width':390,'height':844});assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
 page.locator('#campaign').screenshot(path=str(R/'progress-mobile.png'))
 assert not errors
 print(json.dumps(dict(checks=['separate generated/trained counts','training versus elapsed time','new data replay','missing root value not invented','komi4 final score','mobile width'],errors=errors),indent=2));b.close()
