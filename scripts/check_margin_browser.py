import json
from pathlib import Path
from playwright.sync_api import sync_playwright
R=Path('artifacts/margin-pilot')
with sync_playwright() as p:
 b=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox']);page=b.new_page(viewport={'width':1440,'height':1100});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
 page.goto('http://100.95.30.96:8766/');page.wait_for_function('replay && research && analysisGroups.length>0')
 assert page.locator('#analysis-group option').count()>=43
 assert '6.04' in page.locator('#prediction-metrics').inner_text()
 assert '9勝0分15敗' in page.locator('#match-metrics').inner_text()
 option=page.locator('#analysis-group option').evaluate_all('(xs)=>xs.find(x=>x.textContent.includes("standard-self")).value')
 page.select_option('#analysis-group',option);page.select_option('#komi','4')
 assert '50.00%' in page.locator('#seat-results').inner_text()
 assert '本番推奨は保留' in page.locator('#komi-advice').inner_text()
 page.locator('#analysis').scroll_into_view_if_needed();page.screenshot(path=str(R/'analysis-desktop.png'),full_page=True)
 page.select_option('#group','experiment:paired-00-0');page.wait_for_function('replay && replay.source.includes("paired-00-0")')
 page.locator('#seek').evaluate('(el)=>{el.value=replay.prefix_plies;el.dispatchEvent(new Event("input"));}')
 assert '途中の後付け予測' in page.locator('#point-prediction').inner_text()
 assert '点' in page.locator('#evaluation-value').inner_text()
 original=page.locator('#point-prediction').inner_text();page.fill('#prediction-komi','4');assert page.locator('#point-prediction').inner_text()!=original
 page.select_option('#perspective','1');assert '黄・後手視点' in page.locator('#point-prediction').inner_text()
 page.locator('#seek').evaluate('(el)=>{el.value=el.max;el.dispatchEvent(new Event("input"));}')
 assert '終局の正確値' in page.locator('#point-prediction').inner_text()
 assert page.locator('#result').is_visible()
 page.set_viewport_size({'width':390,'height':844});assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
 page.screenshot(path=str(R/'analysis-mobile.png'),full_page=True)
 assert not errors,errors
 print(json.dumps(dict(errors=errors,checks=['komi 4 regrade 50%','standard initial caveat','test MAE','match results','recorded point search units','posthoc prediction','perspective','komi correction','terminal exact score','mobile width']),indent=2));b.close()
