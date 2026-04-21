/**
 * svg_to_png.js
 * 使用本机 Chrome headless 渲染 SVG → 高清 PNG
 * 优点：系统字体（PingFang SC）完整，scale 2x 清晰度高
 *
 * Usage: node svg_to_png.js <svg_file> <output_png> [scale=2]
 *   or:  node svg_to_png.js --batch  (渲染预定义列表)
 */

const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

// Chrome 路径（macOS）
const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';

// 批量任务配置
const BATCH_TASKS = [
  {
    svg: path.resolve(__dirname, '../src/zh/svg/chapter_llm_06_landscape.svg'),
    out: path.resolve(__dirname, 'images/chapter_llm_06_landscape.png'),
    scale: 2,
  },
  {
    svg: path.resolve(__dirname, '../src/zh/svg/chapter_llm_07_decoder_layer.svg'),
    out: path.resolve(__dirname, 'images/chapter_llm_07_decoder_layer.png'),
    scale: 2,
  },
  {
    svg: path.resolve(__dirname, '../src/zh/svg/chapter_llm_07_attention_evolution.svg'),
    out: path.resolve(__dirname, 'images/chapter_llm_07_attention_evolution.png'),
    scale: 2,
  },
];

async function svgToPng(svgFile, outFile, scale = 2) {
  const svgContent = fs.readFileSync(svgFile, 'utf8');

  // 解析 SVG 原始尺寸
  const widthMatch = svgContent.match(/width="(\d+(?:\.\d+)?)"/);
  const heightMatch = svgContent.match(/height="(\d+(?:\.\d+)?)"/);
  const viewBoxMatch = svgContent.match(/viewBox="[\d.]+ [\d.]+ ([\d.]+) ([\d.]+)"/);

  let svgW = widthMatch ? parseFloat(widthMatch[1]) : (viewBoxMatch ? parseFloat(viewBoxMatch[1]) : 800);
  let svgH = heightMatch ? parseFloat(heightMatch[1]) : (viewBoxMatch ? parseFloat(viewBoxMatch[2]) : 450);

  // 构建 HTML 包装页，添加白色背景
  const html = `<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  html, body {
    width: ${svgW}px;
    height: ${svgH}px;
    overflow: hidden;
    background: #ffffff;
  }
  svg {
    display: block;
    width: ${svgW}px;
    height: ${svgH}px;
  }
</style>
</head>
<body>
${svgContent}
</body>
</html>`;

  const browser = await puppeteer.launch({
    executablePath: CHROME_PATH,
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--disable-gpu',
      '--hide-scrollbars',
      '--disable-web-security',
      '--font-render-hinting=none',  // 保证字体渲染清晰
    ],
    headless: 'new',
  });

  try {
    const page = await browser.newPage();

    // 设置 deviceScaleFactor 实现高分辨率
    await page.setViewport({
      width: svgW,
      height: svgH,
      deviceScaleFactor: scale,
    });

    await page.setContent(html, { waitUntil: 'domcontentloaded' });

    // 等待字体加载
    await page.evaluate(() => document.fonts.ready);

    const outDir = path.dirname(outFile);
    if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });

    await page.screenshot({
      path: outFile,
      type: 'png',
      clip: { x: 0, y: 0, width: svgW, height: svgH },
      omitBackground: false,
    });

    const stat = fs.statSync(outFile);
    console.log(`✅ ${path.basename(outFile)}  ${svgW * scale}×${svgH * scale}px  (${Math.round(stat.size / 1024)}K)`);
  } finally {
    await browser.close();
  }
}

async function main() {
  const args = process.argv.slice(2);

  if (args[0] === '--batch') {
    console.log(`🚀 批量渲染 ${BATCH_TASKS.length} 张 SVG → PNG (scale=${BATCH_TASKS[0].scale}x)\n`);
    for (const task of BATCH_TASKS) {
      await svgToPng(task.svg, task.out, task.scale);
    }
    console.log('\n🎉 全部完成！');
  } else if (args.length >= 2) {
    const [svgFile, outFile, scaleStr] = args;
    const scale = scaleStr ? parseFloat(scaleStr) : 2;
    await svgToPng(path.resolve(svgFile), path.resolve(outFile), scale);
  } else {
    console.log('Usage:');
    console.log('  node svg_to_png.js --batch');
    console.log('  node svg_to_png.js <input.svg> <output.png> [scale=2]');
  }
}

main().catch(err => {
  console.error('❌ Error:', err.message);
  process.exit(1);
});
