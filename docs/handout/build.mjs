// Renders a handout Markdown file into the single-page HTML format used in
// 26秋招/面试讲义: fixed left table of contents (## and ### headings), the
// current section highlighted while scrolling, collapsed into a button on
// narrow screens. Images written as ![caption](img/x.svg) become click-to-open
// figures.
//   node docs/handout/build.mjs <in.md> <out.html>
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import MarkdownIt from "markdown-it";

const [, , inPath, outPath] = process.argv;
if (!inPath || !outPath) {
  console.error("usage: node build.mjs <in.md> <out.html>");
  process.exit(2);
}
const here = path.dirname(fileURLToPath(import.meta.url));
const css = fs.readFileSync(path.join(here, "style.css"), "utf8");
const src = fs.readFileSync(inPath, "utf8");

const md = new MarkdownIt({ html: true, linkify: false, typographer: false });
const toc = [];
let title = "讲义";
let h2 = 0, h3 = 0;

md.core.ruler.push("ids", (state) => {
  const t = state.tokens;
  for (let i = 0; i < t.length; i++) {
    if (t[i].type !== "heading_open") continue;
    const text = t[i + 1].content;
    if (t[i].tag === "h1") { title = text; continue; }
    if (t[i].tag === "h2") { h3 = 0; const id = `s${h2++}`; t[i].attrSet("id", id); toc.push({ lvl: 1, id, text }); }
    if (t[i].tag === "h3") { const id = `s${h2 - 1}-${h3++}`; t[i].attrSet("id", id); toc.push({ lvl: 2, id, text }); }
  }
});

md.renderer.rules.image = (tokens, idx) => {
  const tok = tokens[idx];
  const src = tok.attrGet("src");
  const alt = tok.content;
  return `</p><figure class="fig"><a href="${src}" target="_blank" title="点击在新标签页看原图"><img alt="${md.utils.escapeHtml(alt)}" src="${src}" /></a><figcaption>${md.utils.escapeHtml(alt)} · 点图可在新标签页放大</figcaption></figure><p>`;
};

const body = md.render(src).replace(/<p><\/p>/g, "");
const esc = md.utils.escapeHtml;
const aside = toc.map((e) => `<a class="l${e.lvl}" href="#${e.id}">${esc(e.text)}</a>`).join("");

const script = `(function(){
  var links=[].slice.call(document.querySelectorAll('aside a.l1, aside a.l2'));
  var heads=links.map(function(a){return document.getElementById(a.getAttribute('href').slice(1));});
  function onScroll(){
    var y=window.scrollY+80, cur=0;
    heads.forEach(function(h,i){ if(h && h.offsetTop<=y) cur=i; });
    links.forEach(function(a,i){ a.classList.toggle('on', i===cur); });
  }
  window.addEventListener('scroll', onScroll, {passive:true}); onScroll();
  document.getElementById('toggle').addEventListener('click', function(){ document.body.classList.toggle('open'); });
  links.forEach(function(a){ a.addEventListener('click', function(){ document.body.classList.remove('open'); }); });
})();`;

const html = `<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>${esc(title)}</title><style>
${css}</style></head><body><button id="toggle" type="button">目录</button><aside aria-label="目录"><a class="doc-title" href="#top">${esc(title)}</a>${aside}</aside><main><div class="wrap" id="top">
${body}</div></main><script>
${script}
</script></body></html>
`;
fs.writeFileSync(outPath, html);
console.log(`wrote ${outPath} (${toc.length} toc entries)`);
