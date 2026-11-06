// Browser interaction regression; PLAYWRIGHT_MODULE may point to an isolated install.
const {chromium}=require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const fs=require('fs');
const assert=require('assert/strict');
(async()=>{
 const url=process.argv[2]||'http://127.0.0.1:8766', out=process.argv[3]||'/tmp/deep-web-browser';fs.mkdirSync(out,{recursive:true});fs.rmSync(out+'/result.json',{force:true});
 const browser=await chromium.launch({headless:true,args:['--no-sandbox']});const page=await browser.newPage({viewport:{width:1100,height:900},acceptDownloads:true});const errors=[];page.on('pageerror',e=>errors.push(String(e)));page.on('dialog',d=>d.accept());
 async function state(){return page.evaluate(()=>game)}
 async function ready(mode,seat){await page.waitForFunction(([m,s])=>game?.mode===m&&game.seat===s&&!game.busy&&!waiting&&!game.error,[mode,seat]);}
 async function newGame(mode,seat){await page.selectOption('#mode',mode);await page.selectOption('#seat',String(seat));await page.click('#new');await ready(mode,seat);}
 async function choose(){
  const q=await page.evaluate(()=>game.legal.find(q=>q[0]===piece&&q[1]===orientation));assert(q);
  await page.getByRole('button',{name:new RegExp(`^${q[2]+1}行${q[3]+1}列(?: |$)`)}).click();assert(await page.locator('#place').isEnabled());return q;
 }
 try{
  await page.goto(url);await ready('policy',0);assert.equal((await state()).generation,'gen3');assert(await page.locator('#budget').isDisabled());
  await newGame('deep',0);let g=await state();assert.equal(g.budget,500);assert.equal(g.model_seed,1751401);assert.equal(g.generation,'gen3');assert.deepEqual(await page.locator('#budget option').evaluateAll(xs=>xs.map(x=>x.value)),['500']);assert(g.legal.length>0);assert(await page.locator('#board circle').count()>0);
  const asymmetric=await page.evaluate(()=>game.remaining[game.seat].find(id=>pieces[id].length===8));await page.getByRole('button',{name:new RegExp(`^駒${asymmetric+1}、`)}).click();
  let ori=await page.evaluate(()=>orientation);await page.click('#rotate');assert.notEqual(await page.evaluate(()=>orientation),ori);ori=await page.evaluate(()=>orientation);await page.click('#flip');assert.notEqual(await page.evaluate(()=>orientation),ori);
  await choose();await page.click('#place');await page.waitForFunction(()=>game.revision>=2&&!game.busy&&!waiting);g=await state();assert.equal(g.turn,0);assert(g.stats.elapsed_ms>=0);assert([2,4,6].includes(g.stats.depth)||g.stats.fallback);
  const download=page.waitForEvent('download');await page.click('#download');const file=await download;const saved=JSON.parse(fs.readFileSync(await file.path(),'utf8'));assert.equal(saved.generation,'gen3');assert.equal(saved.model_seed,1751401);assert(saved.history.length>=2);assert(!('id' in saved));fs.writeFileSync(out+'/download.json',JSON.stringify(saved,null,2));
  await page.screenshot({path:out+'/linear-first.png',fullPage:true});
  await newGame('deep',1);g=await state();assert(g.history.length>=1&&g.turn===1&&g.legal.length>0);await choose();await page.click('#place');await page.waitForFunction(()=>game.revision>=3&&!game.busy&&!waiting);await page.screenshot({path:out+'/linear-second.png',fullPage:true});
  // Complete a human game using visible controls, then save terminal scores/passes.
  let plies=0;
  while(!(await state()).done){
   await page.waitForFunction(()=>!game.busy&&!waiting);if((await state()).done)break;
   const q=await page.evaluate(()=>game.legal.find(q=>q[1]===0)||game.legal[0]);assert(q);
   await page.getByRole('button',{name:new RegExp(`^駒${q[0]+1}、`)}).click();
   let found=false;
   for(let flip=0;flip<2&&!found;flip++){
    for(let rot=0;rot<4;rot++){
     if(await page.evaluate(o=>orientation===o,q[1])){found=true;break;}
     await page.click('#rotate');
    }
    if(!found)await page.click('#flip');
   }
   assert(found);assert.equal(await page.locator('#board circle').count(),await page.evaluate(()=>new Set(game.legal.filter(q=>q[0]===piece&&q[1]===orientation).map(q=>q[2]*14+q[3])).size));await page.getByRole('button',{name:new RegExp(`^${q[2]+1}行${q[3]+1}列(?: |$)`)}).click();await page.click('#place');await page.waitForFunction(()=>!game.busy&&!waiting);assert(++plies<43);
  }
  const terminal=await state();assert(terminal.done);assert(terminal.history.includes(null));assert.match(await page.locator('#status').textContent(),/勝ち|引き分け/);
  const doneDownload=page.waitForEvent('download');await page.click('#download');const doneFile=await doneDownload;const doneSaved=JSON.parse(fs.readFileSync(await doneFile.path(),'utf8'));assert.deepEqual(doneSaved.scores,terminal.scores);assert(doneSaved.done);assert.equal(doneSaved.move_stats.length,doneSaved.history.length);assert(doneSaved.move_stats.some(x=>x&&[2,4,6].includes(x.depth)&&x.elapsed_ms>=0));fs.writeFileSync(out+'/terminal-download.json',JSON.stringify(doneSaved,null,2));
  await newGame('deep',1);
  // Keep an old move response in flight while the user opens a new game.
  await page.route('**/api/move',async route=>{const response=await route.fetch();await new Promise(r=>setTimeout(r,700));await route.fulfill({response})});
  await choose();await page.click('#place');await newGame('policy',0);const newId=(await state()).id;await page.waitForTimeout(1000);g=await state();assert.equal(g.id,newId);assert.equal(g.mode,'policy');assert.equal(g.revision,0);assert.equal(await page.locator('#error').textContent(),'');await page.unroute('**/api/move');
  await page.selectOption('#mode','search');assert(await page.locator('#budget').isEnabled());await page.selectOption('#budget','100');await page.selectOption('#seat','1');await page.click('#new');await ready('search',1);assert((await state()).history.length>=1);
  // A separate browser context gets an isolated game.
  const context=await browser.newContext();const other=await context.newPage();await other.goto(url);await other.waitForFunction(()=>game&&!game.busy);assert.notEqual(await other.evaluate(()=>game.id),(await state()).id);await context.close();
  await newGame('deep',0);await page.setViewportSize({width:390,height:844});assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));await page.screenshot({path:out+'/mobile.png',fullPage:true});
  assert.deepEqual(errors,[]);fs.writeFileSync(out+'/result.json',JSON.stringify({pass:true,url,default_gen3:true,deep_both_seats:true,rotation_flip:true,legal_anchors:true,ai_moves:true,mode_switch:true,stale_response_discard:true,download_metadata:true,terminal_pass_and_scores:true,session_isolation:true,page_errors:errors},null,2));
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exit(1)});
