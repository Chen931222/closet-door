/**
 * ASUS UGen AI League Stage I：Closet Door 展演影片的實機操作段。
 *
 *   （先開 closet-door 的本機伺服器：127.0.0.1:8765）
 *   node closet-door.mjs
 *
 * 產出 G:\Projects\closet-door\video\raw\*.webm，由 closet-door/video/build.sh 接卡片、轉 mp4。
 * 字幕列放上方：下方是「已經有的／今日穿搭」兩欄，蓋住就看不到結果。
 */
import { chromium } from 'playwright-core';
import { mkdirSync } from 'node:fs';
import * as k from './lib.mjs';

const CHROME = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const BASE = 'http://127.0.0.1:8765/';
const OUT_DIR = 'G:\\Projects\\closet-door\\video\\raw';
mkdirSync(OUT_DIR, { recursive: true });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const CAPTION_INIT = `
  (function(){
    function ensure(){
      let el = document.getElementById('__cap');
      if (el) return el;
      el = document.createElement('div');
      el.id = '__cap';
      el.style.cssText = 'position:fixed;left:0;right:0;top:0;z-index:2147483646;padding:16px 32px 17px;background:rgba(29,27,24,.92);color:#f4f1ea;font:21px/1.4 Georgia,serif;opacity:0;transition:opacity .35s;pointer-events:none;border-bottom:3px solid #9a3b2b';
      (document.body || document.documentElement).appendChild(el);
      return el;
    }
    window.__cap = function(t){ const el = ensure(); el.textContent = t; el.style.opacity = t ? '1' : '0'; };
  })();
`;
const cap = (page, t) => page.evaluate((x) => window.__cap(x), t);
// 字幕只講畫面上真的出現的結果：先讀 Read-on-device 表格
const facts = (page) => page.evaluate(() => Object.fromEntries([...document.querySelectorAll('#facts tr')].map((tr) => [tr.cells[0].textContent.trim(), tr.cells[1].textContent.trim()])));
function expect(cond, msg) { if (!cond) throw new Error('畫面跟字幕對不上：' + msg); }

async function center(page, selector, nth = 0) {
  return page.evaluate(([sel, n]) => {
    const el = document.querySelectorAll(sel)[n];
    el.scrollIntoView({ block: 'nearest' });
    const r = el.getBoundingClientRect();
    return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
  }, [selector, nth]);
}

async function pickDoor(page, nth) {
  const p = await center(page, '.door img', nth);
  await k.moveTo(page, p.x, p.y, 700);
  await sleep(250);
  const done = page.waitForResponse((r) => r.url().endsWith('/analyze'));
  await k.click(page, { settle: 0 });
  await done;
  await sleep(700);
}

async function pointAt(page, selector, nth = 0) {
  const p = await center(page, selector, nth);
  await k.moveTo(page, p.x, p.y, 700);
}

const browser = await chromium.launch({ executablePath: CHROME, headless: true });
const ctx = await browser.newContext({
  viewport: { width: 1280, height: 720 }, deviceScaleFactor: 1, locale: 'en-US',
  recordVideo: { dir: OUT_DIR, size: { width: 1280, height: 720 } },
});
await ctx.addInitScript(CAPTION_INIT);
const page = await ctx.newPage();
await page.goto(BASE);
await page.waitForResponse((r) => r.url().endsWith('/analyze'));
await page.waitForFunction(() => [...document.querySelectorAll('.door img')].slice(0, 8).every((i) => i.complete && i.naturalWidth));
await sleep(600);
await k.installCursor(page);
await page.evaluate(() => { window.__cur(640, 360); window.__lastCur = { x: 640, y: 360 }; });

await cap(page, 'Stage I prototype, running locally. Left: photos taken on my closet door. Every garment I own was shot hanging there.');
await sleep(4200);
await pointAt(page, '#cut');
await cap(page, 'On the device: U2Net-p cuts the garment out, CLIP (int8) names its type, k-means finds the main colour.');
await sleep(4600);
await pointAt(page, '#facts td', 1);
await sleep(1400);
await pointAt(page, '#similar .row');
await cap(page, '"You already own this": the closest items in the same slot. This shirt matches its own entry in my closet at 0.93.');
await sleep(5200);
await pointAt(page, '#outfits .row');
await cap(page, 'Outfits with it for today: warmth vs temperature, at most one accent colour, light/dark contrast. Each lists its reasons.');
await sleep(5600);

await cap(page, 'A white track jacket.');
await pickDoor(page, 7);
const j = await facts(page);
await cap(page, `Read as ${j['Slot']} (${j['Type']}), main colour "${j['Main colour']}"` + (['white', 'light gray', 'cream'].includes(j['Main colour']) ? '.' : ' — it is white; warm light again.') + ' Similar pieces I already own appear below.');
await sleep(5200);

await cap(page, 'Now colder: 12 °C.');
const s = await center(page, '#temp');
await k.moveTo(page, s.x, s.y, 600);
await page.evaluate(() => { const t = document.querySelector('#temp'); t.value = 12; t.dispatchEvent(new Event('input')); t.dispatchEvent(new Event('change')); });
await page.waitForResponse((r) => r.url().endsWith('/analyze'));
await sleep(600);
await pointAt(page, '#outfits .row');
await cap(page, 'The suggestions re-rank for 12 °C: warmer pieces win. No cloud call; the closet index is two files on this machine.');
await sleep(5600);

await cap(page, 'What does not work yet. Shorts hanging on the door...');
await pickDoor(page, 2);
const sh = await facts(page);
expect(sh['Slot'] !== 'bottom', 'shorts slot = ' + sh['Slot']);
await pointAt(page, '#facts td', 1);
await cap(page, '...are read as a shirt. Hanging shorts were wrong in 2 of the first 8 door photos I checked. Stage II: more garment prompts and a hand-labelled door set.');
await sleep(6400);

await cap(page, 'Light-blue jeans under warm room light...');
await pickDoor(page, 5);
const jn = await facts(page);
expect(jn['Main colour'] === 'olive', 'jeans colour = ' + jn['Main colour']);
await pointAt(page, '#facts td', 7);
await cap(page, '...come out "olive". A phone sees different light every time; a fixed door camera does not. Stage II adds a white reference card on the door.');
await sleep(6600);

await pointAt(page, '#times');
await cap(page, 'Time per stage on a desktop CPU, int8: about 0.16 s per garment on 4 threads. The cutout is the slow part, so it moves to the Hailo-10H first.');
await sleep(6400);
await cap(page, '');
await sleep(800);

const v = page.video();
await ctx.close();
await browser.close();
console.log('saved', await v.path());
