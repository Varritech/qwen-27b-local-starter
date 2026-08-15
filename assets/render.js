const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({
    viewport: { width: 1200, height: 1200 }
  });
  
  await page.goto(`file://${process.argv[2]}`, { waitUntil: 'networkidle' });
  await page.screenshot({ 
    path: process.argv[3],
    type: 'png',
    scale: 'device'
  });
  
  await browser.close();
  console.log('✅ Image rendered successfully');
})();
