const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch({ channel: 'chrome', headless: true });
  const page = await browser.newPage({ viewport: { width: 1700, height: 1300 } });
  const errors = [];
  page.on('console', m => { if (m.type()==='error'||m.type()==='warning') errors.push(`[${m.type()}] ${m.text()}`); });
  page.on('pageerror', e => errors.push(`[pageerror] ${e.message}`));
  await page.goto('http://127.0.0.1:8002/', { waitUntil: 'load' });
  await page.waitForTimeout(3500);

  const info = await page.evaluate(() => {
    const rows = [...document.querySelectorAll('#species-body tr')];
    const firstCells = rows.length ? [...rows[0].querySelectorAll('td')].map(td => td.textContent) : [];
    // count how many cells are empty/undefined/NaN across all rows
    let bad = 0, total = 0;
    for (const r of rows) for (const td of r.querySelectorAll('td')) {
      total++;
      const t = td.textContent.trim();
      if (t === '' || t === 'undefined' || t === 'NaN') bad++;
    }
    return { rowCount: rows.length, firstCells, badCells: bad, totalCells: total,
             bodiesDrawn: (typeof gpuReady !== 'undefined') ? gpuReady : null };
  });
  console.log('console issues:', JSON.stringify(errors));
  console.log('species rows rendered:', info.rowCount);
  console.log('first row cells:', JSON.stringify(info.firstCells));
  console.log('empty/undefined/NaN cells:', info.badCells, 'of', info.totalCells);
  console.log('gpuReady:', info.bodiesDrawn);
  await page.screenshot({ path: 'screenshot_species_opt.png' });
  await browser.close();
})().catch(e => { console.error('FATAL', e); process.exit(1); });
