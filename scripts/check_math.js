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

// Markdown is processed BEFORE the math renderer sees the source, so a literal
// `*` inside an expression is consumed as emphasis and the LaTeX that reaches
// KaTeX is not the LaTeX in the file. `$h^*_i$` arrives as `$h^_i$` and fails
// with "Missing open brace for superscript" - on the rendered page only.
//
// This checker used to validate the raw source and passed while the published
// page showed a red error box. Checking the source alone is not enough: the
// hazard has to be caught here, because nothing downstream will.
const MD_HAZARDS = [
  [/\*/, 'a literal * (markdown eats it as emphasis) - use \\ast, \\star or \\cdot'],
  [/(?<!\\)~/, 'a literal ~ (markdown strikethrough) - use \\sim or \\approx'],
];

// CommonMark: "Any ASCII punctuation character may be backslash-escaped." The
// markdown pass therefore strips the backslash from \; \, \! \{ \} \\ and KaTeX
// never sees the command - `a \; = \; b` reaches it as `a ; = ; b`, which is why
// stray semicolons appeared in every equation. Letter-based commands (\quad,
// \lbrace, \cdot) are unaffected, so those are the substitutes.
const PUNCT_ESCAPES = /\\([!"#$%&'()*+,\-./:;<=>?@\[\]^_`{|}~\\])/;
const PUNCT_FIX = {
  ';': 'drop it, or use \\quad',
  ',': 'drop it, or use \\quad',
  ':': 'drop it, or use \\quad',
  '!': 'drop it',
  '{': 'use \\lbrace',
  '}': 'use \\rbrace',
  '\\': 'restructure so no line break is needed',
};

// GitHub does not run stock KaTeX: it runs it with a restricted macro
// allowlist, and rejects anything that can define or inject. `\operatorname` is
// on that list. Locally-installed KaTeX accepts all of these happily, so this
// checker passed on markup GitHub refused with
//   "The following macros are not allowed: operatorname"
// which is the same failure mode as the markdown-emphasis hazard above: the
// renderer that matters is stricter than the one being tested against.
const DENIED_MACROS = [
  'operatorname', 'newcommand', 'renewcommand', 'providecommand',
  'def', 'gdef', 'edef', 'let', 'includegraphics', 'href', 'url',
  'htmlClass', 'htmlId', 'htmlStyle', 'htmlData', 'input', 'include',
];
const SUBSTITUTE = { operatorname: '\\mathrm' };

function deniedMacro(tex) {
  for (const mac of DENIED_MACROS) {
    if (new RegExp('\\\\' + mac + '(?![a-zA-Z])').test(tex)) return mac;
  }
  return null;
}

function hazards(tex, label, i) {
  for (const pair of MD_HAZARDS) {
    if (pair[0].test(tex)) {
      problems.push(label + ' #' + (i + 1) + ': contains ' + pair[1] +
                    '\n    ' + tex.trim().slice(0, 150));
      return true;
    }
  }
  const esc = PUNCT_ESCAPES.exec(tex);
  if (esc) {
    const fix = PUNCT_FIX[esc[1]] || 'use a letter-based command instead';
    problems.push(label + ' #' + (i + 1) + ': contains \\' + esc[1] +
                  ' - markdown strips the backslash before KaTeX sees it; ' + fix +
                  '\n    ' + tex.trim().slice(0, 150));
    return true;
  }

  const mac = deniedMacro(tex);
  if (mac) {
    const fix = SUBSTITUTE[mac] ? ' - use ' + SUBSTITUTE[mac] + ' instead' : '';
    problems.push(label + ' #' + (i + 1) + ': uses \\' + mac +
                  ', which GitHub\'s KaTeX does not allow' + fix +
                  '\n    ' + tex.trim().slice(0, 150));
    return true;
  }
  return false;
}

function run(list, label) {
  list.forEach((tex, i) => {
    const seen = [];
    if (hazards(tex, label, i)) return;
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
