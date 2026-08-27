// Render every equation in README.md through KaTeX, which is what GitHub uses.
//
//   npm install katex      (once, anywhere on the path)
//   node scripts/check_math.js [file.md]
//
// A block that throws here does not fail quietly on GitHub - it renders as red
// error text in the middle of the document. Markdown has no compiler, so this
// is the only thing standing between a mistyped brace and a broken page.
//
// Warnings are treated as failures too. The usual one is a literal Unicode
// symbol (σ, α) inside \text{}: KaTeX has no metrics for it, so it renders in a
// fallback font at the wrong size next to a properly typeset \sigma.
const fs = require('fs');
const path = require('path');

let katex;
try {
  katex = require('katex');
} catch (e) {
  console.log('skip: katex is not installed (npm install katex)');
  process.exit(0);
}

const file = process.argv[2] || path.resolve(__dirname, '..', 'README.md');
const src = fs.readFileSync(file, 'utf8');

const display = [...src.matchAll(/\$\$([\s\S]+?)\$\$/g)].map(m => m[1]);

// Inline math: $...$ within one line, skipping lines that carry a $$ block.
const inline = [];
for (const line of src.split('\n')) {
  if (line.includes('$$')) continue;
  for (const m of line.matchAll(/(?<!\$)\$([^$\n]+?)\$(?!\$)/g)) inline.push(m[1]);
}

const problems = [];
const warnings = [];

function run(list, label) {
  list.forEach((tex, i) => {
    const seen = [];
    try {
      katex.renderToString(tex, {
        displayMode: label === 'display',
        throwOnError: true,
        strict: (code, msg) => { seen.push(`${code}: ${msg}`); return 'ignore'; },
      });
    } catch (e) {
      problems.push(`${label} #${i + 1}: ${e.message}\n    ${tex.trim().slice(0, 150)}`);
      return;
    }
    if (seen.length) {
      warnings.push(`${label} #${i + 1}: ${seen[0]}\n    ${tex.trim().slice(0, 150)}`);
    }
  });
  console.log(`  ${label}: ${list.length} checked`);
}

console.log(`checking ${path.basename(file)}`);
run(display, 'display');
run(inline, 'inline');

for (const w of warnings) console.error('\nWARNING  ' + w);
for (const p of problems) console.error('\nERROR    ' + p);

if (problems.length || warnings.length) {
  console.error(`\n${problems.length} error(s), ${warnings.length} warning(s)`);
  process.exit(1);
}
console.log('\nall math renders cleanly under KaTeX');
