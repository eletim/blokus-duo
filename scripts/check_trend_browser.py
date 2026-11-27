import json
from pathlib import Path
from playwright.sync_api import sync_playwright
R=Path('artifacts/komi4-trend')
with sync_playwright() as p:
    b=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox']);page=b.new_page(viewport={'width':1440,'height':1000});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto('http://100.95.30.96:8766/');page.wait_for_function('trend?.plan')
    assert page.evaluate('trend.plan.additional_unique_train_target')==22000
    assert page.locator('#trend-rows tr').count()==page.evaluate('trend.benchmarks.length')
    assert '純GPU' in page.locator('#trend-time').inner_text()
    page.select_option('#trend-metric','point_rate');page.select_option('#trend-seat','1');page.select_option('#trend-opponent','prior')
    if page.evaluate('trend.benchmarks.length'):
        assert page.locator('#trend-chart circle').count()==page.evaluate('trend.benchmarks.length')
        page.click('#trend-watch');page.wait_for_function('replay?.players.some(p=>p.includes("固定条件点差"))')
    page.locator('#trend').screenshot(path=str(R/'progress-desktop.png'))
    page.set_viewport_size({'width':390,'height':844});assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
    page.locator('#trend').screenshot(path=str(R/'progress-mobile.png'));assert not errors
    result=dict(evaluated_generations=page.evaluate('trend.benchmarks.length'),errors=errors,checks=['real plan and counts','only completed generations','metric/seat/opponent controls','correct training-validation time label','mobile width'])
    (R/'browser-progress.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(result);b.close()
