import json
from pathlib import Path
from playwright.sync_api import sync_playwright
R=Path('artifacts/komi4-campaign')
with sync_playwright() as p:
 b=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox']);page=b.new_page(viewport={'width':1440,'height':1100});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
 page.goto('http://100.95.30.96:8766/');page.wait_for_function('campaign?.results && replay')
 assert page.evaluate('campaign.results.completed_games')==188
 assert page.locator('#campaign-matches tr').count()==4
 assert '4.28' in page.locator('#campaign-prediction').inner_text()
 assert '推定不十分' in page.locator('#campaign-matches').inner_text()
 assert '2 seedは更新前モデルを保持' in page.locator('#campaign-limits').inner_text()
 page.locator('#campaign').screenshot(path=str(R/'campaign-panel.png'))
 page.click('#campaign-watch');page.wait_for_function('replay?.source.includes("comparison")')
 page.locator('#seek').evaluate('(el)=>{el.value=replay.prefix_plies;el.dispatchEvent(new Event("input"));}')
 assert '世代1' in page.locator('#point-prediction').inner_text()
 assert '後手+4点' in page.locator('#point-prediction').inner_text()
 page.set_viewport_size({'width':390,'height':844});assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
 page.locator('#campaign').screenshot(path=str(R/'campaign-panel-mobile.png'))
 assert not errors
 print(json.dumps(dict(completed_games=188,rows=4,checks=['test MAE','small-sample caveat','retained bootstrap disclosure','generation1 replay link','mobile width'],browser_errors=errors),ensure_ascii=False,indent=2));b.close()
