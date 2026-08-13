import assert from "node:assert/strict";
import { existsSync, lstatSync, readdirSync, readFileSync } from "node:fs";
import { test } from "node:test";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { GUIDES, MAX_INLINE_DEPTH, MAX_INLINE_LENGTH, MAX_LINE_LENGTH, MAX_SOURCE_BYTES, SUPPORTED_MARKDOWN, checkOrWrite, headingSlug, renderInline, renderMarkdown, resolveGuideHref } from "./build-guides.mjs";

const docs = dirname(fileURLToPath(import.meta.url));
const index = readFileSync(resolve(docs, "index.html"), "utf8");
const guideFiles = readdirSync(resolve(docs, "guides"));
const styles = readFileSync(resolve(docs, "styles.css"), "utf8");

const decodeHtml = (value) => value.replace(/&(amp|lt|gt|quot);/g, (_, name) => ({ amp: "&", lt: "<", gt: ">", quot: '"' })[name]);
const words = (value) => (value.toLowerCase().match(/[\p{L}\p{N}_-]+/gu) ?? []).filter((word) => /[\p{L}\p{N}]/u.test(word));
const visibleSource = (markdown) => markdown
  .replace(/!?\[([^\]]*)\]\([^)]*\)/g, "$1")
  .replace(/^```.*$/gm, "")
  .replace(/^[#>*+\-]+\s*/gm, "")
  .replace(/^\d+\.\s*/gm, "")
  .replace(/[`*~|]/g, " ");

function assertTextCoverage(markdown, guide, source) {
  const article = guide.match(/<article class="guide-article">([\s\S]+)<\/article>/)?.[1] ?? "";
  const outputCounts = new Map();
  for (const word of words(decodeHtml(article.replace(/<[^>]+>/g, " ")))) outputCounts.set(word, (outputCounts.get(word) ?? 0) + 1);
  const sourceCounts = new Map();
  for (const word of words(visibleSource(markdown))) sourceCounts.set(word, (sourceCounts.get(word) ?? 0) + 1);
  for (const [word, count] of sourceCounts) assert.ok((outputCounts.get(word) ?? 0) >= count, `${source} preserves “${word}” text`);
}

test("every Markdown source has a checked-in rendered guide", () => {
  const sources = readdirSync(docs).filter((name) => name.endsWith(".md")).sort();
  assert.deepEqual(Object.keys(GUIDES).sort(), sources);
  for (const [source, output] of Object.entries(GUIDES)) {
    const guidePath = resolve(docs, "guides", output);
    assert.ok(existsSync(guidePath), `${output} is checked in`);
    assert.ok(lstatSync(guidePath).isFile() && !lstatSync(guidePath).isSymbolicLink(), `${output} is a regular file`);
    const guide = readFileSync(guidePath, "utf8");
    assert.match(guide, /^<!doctype html>/i);
    assert.match(guide, /href="\.\.\/styles\.css"/);
    assert.match(guide, /<article class="guide-article">/);
    assert.match(guide, /<main id="guide-content" class="guide-main" tabindex="-1">/);
    assert.match(guide, new RegExp(`Guide source: <code>${source.replace(".", "\\.")}</code>`));
    assert.match(guide, /<h1 id="[^"]+">/);
    assert.doesNotMatch(guide, /href="(?!https?:\/\/)[^"#]*\.md(?:[?#][^"]*)?"/);
    assert.doesNotMatch(guide, /href="(?:javascript|data):/i);
    assert.doesNotMatch(guide, /<script\b/i);

    const markdown = readFileSync(resolve(docs, source), "utf8");
    const headingCounts = new Map();
    for (const match of markdown.matchAll(/^(#{2,3})\s+(.+?)\s*#*\s*$/gm)) {
      const baseSlug = headingSlug(match[2]);
      const count = headingCounts.get(baseSlug) ?? 0;
      headingCounts.set(baseSlug, count + 1);
      const id = count ? `${baseSlug}-${count}` : baseSlug;
      assert.ok(guide.includes(`<h${match[1].length} id="${id}">`), `${source} renders heading ${match[2]}`);
    }
    assertTextCoverage(markdown, guide, source);

    const tableCount = (markdown.match(/^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*$/gm) ?? []).length;
    assert.equal((guide.match(/class="guide-table-wrap" tabindex="0" role="region" aria-label="Documentation table"/g) ?? []).length, tableCount);
  }
  assert.deepEqual(guideFiles.sort(), Object.values(GUIDES).sort());
  assert.match(styles, /\.guide-article th\{color:#596477\}/);
});

test("index links only to checked-in local HTML and keeps external Markdown external", () => {
  const hrefs = [...index.matchAll(/href="([^"]+)"/g)].map((match) => match[1]);
  assert.equal(hrefs.filter((href) => !/^[a-z][a-z0-9+.-]*:/i.test(href) && href.endsWith(".md")).length, 0);
  for (const href of hrefs.filter((href) => href.startsWith("guides/"))) assert.ok(existsSync(resolve(docs, href)), `${href} exists`);
  assert.ok(hrefs.includes("https://github.com/tyhuang9/llama-harness/blob/main/protocol/compatibility/v1.md"));
});

test("checked-in guides exactly match the deterministic renderer", () => {
  assert.doesNotThrow(() => checkOrWrite("--check"));
  const embedding = readFileSync(resolve(docs, "guides", "embedding.html"), "utf8");
  assert.match(embedding, /https:\/\/github\.com\/tyhuang9\/llama-harness\/tree\/main\/examples\/local-task-agent/);
  assert.match(embedding, /href="tauri\.html"/);
});

test("internal rendered-guide fragments resolve to generated heading IDs", () => {
  assert.equal(resolveGuideHref("./architecture.md#runtime-choices", "test.md"), "architecture.html#runtime-choices");
  for (const output of Object.values(GUIDES)) {
    const guide = readFileSync(resolve(docs, "guides", output), "utf8");
    for (const link of guide.matchAll(/href="([^"?#]+\.html)(?:\?[^"#]*)?#([^"]+)"/g)) {
      const target = readFileSync(resolve(docs, "guides", link[1]), "utf8");
      assert.ok(target.includes(`id="${decodeURIComponent(link[2])}"`), `${output} fragment ${link[2]} exists in ${link[1]}`);
    }
  }
});

test("renderer documents and handles its supported flat Markdown subset", () => {
  assert.deepEqual(SUPPORTED_MARKDOWN, [
    "h1-h6 headings", "paragraphs", "fenced code", "flat ordered and unordered lists", "blockquotes",
    "pipe tables", "inline code", "strong and emphasis", "safe links",
  ]);
  const rendered = renderMarkdown(`# Title
## Repeat
## Repeat
Paragraph with \`code\`, **strong**, and *emphasis*.

> Quoted text

- first
- second

1. one
2. two

| Name | Value |
| --- | --- |
| safe | yes |

\`\`\`html
<safe-example>
\`\`\``);
  assert.match(rendered, /<h1 id="title">Title<\/h1>/);
  assert.match(rendered, /<h2 id="repeat">Repeat<\/h2>/);
  assert.match(rendered, /<h2 id="repeat-1">Repeat<\/h2>/);
  assert.match(rendered, /<code>code<\/code>.*<strong>strong<\/strong>.*<em>emphasis<\/em>/);
  assert.match(rendered, /<blockquote><p>Quoted text<\/p><\/blockquote>/);
  assert.match(rendered, /<ul><li>first<\/li><li>second<\/li><\/ul>/);
  assert.match(rendered, /<ol><li>one<\/li><li>two<\/li><\/ol>/);
  assert.match(rendered, /class="guide-table-wrap" tabindex="0" role="region" aria-label="Documentation table"/);
  assert.match(rendered, /<pre><code class="language-html">&lt;safe-example&gt;<\/code><\/pre>/);
  assert.throws(() => renderMarkdown("- parent\n  - child"), /Nested or indented lists are not supported/);
  assert.equal(renderMarkdown("<script>alert('escaped')</script>"), "<p>&lt;script&gt;alert('escaped')&lt;/script&gt;</p>");
});

test("link resolution rejects unsafe forms and canonicalizes repository paths", () => {
  assert.equal(resolveGuideHref("https://example.com/docs", "test.md"), "https://example.com/docs");
  assert.equal(resolveGuideHref("mailto:docs@example.com", "test.md"), "mailto:docs@example.com");
  assert.equal(resolveGuideHref("#local", "test.md"), "#local");
  assert.equal(resolveGuideHref("../README.md#usage", "test.md"), "https://github.com/tyhuang9/llama-harness/blob/main/README.md#usage");
  assert.equal(resolveGuideHref("../examples/local-task-agent", "test.md"), "https://github.com/tyhuang9/llama-harness/tree/main/examples/local-task-agent");
  for (const unsafe of ["javascript:alert(1)", "data:text/html,test", "java&#x73;cript:alert(1)", "javascript&#58;alert(1)", "java\u0000script:alert(1)", "//example.com/path", "\\\\server\\share\\file.md"]) {
    assert.throws(() => resolveGuideHref(unsafe, "test.md"), /Unsafe link protocol|Protocol-relative or UNC link/);
  }
  assert.throws(() => resolveGuideHref("../../", "test.md"), /escapes the repository/);
  assert.throws(() => resolveGuideHref("C:\\Windows\\system.ini", "test.md"), /Absolute local link/);
  assert.throws(() => resolveGuideHref("missing.md", "test.md"), /Unresolved local link/);
});

test("parser bounds adversarial input and rejects malformed fences", { timeout: 1_000 }, () => {
  assert.throws(() => renderMarkdown("x".repeat(MAX_LINE_LENGTH + 1), "long-line.md"), /line exceeds/);
  assert.throws(() => renderMarkdown("x\n".repeat(Math.floor(MAX_SOURCE_BYTES / 2) + 1), "large-source.md"), /source exceeds/);
  assert.throws(() => renderMarkdown("```js\nconst unfinished = true;", "bad-fence.md"), /Unclosed code fence/);
  assert.equal(renderMarkdown("Paragraph after validation.", "progress.md"), "<p>Paragraph after validation.</p>");
});

test("inline rendering is bounded before regex matching and paragraph joining", { timeout: 1_000 }, () => {
  const adversarialParts = Array.from({ length: 80 }, () => "[a(".repeat(20));
  assert.ok(adversarialParts.every((part) => part.length < MAX_LINE_LENGTH));
  assert.throws(() => renderMarkdown(adversarialParts.join("\n"), "adversarial-inline.md"), /Inline Markdown exceeds 4096 characters/);
  assert.throws(() => renderInline("x".repeat(MAX_INLINE_LENGTH + 1), "large-inline.md"), /Inline Markdown exceeds 4096 characters/);
  assert.throws(() => renderInline("normal", "deep-inline.md", MAX_INLINE_DEPTH + 1), /nesting depth 16/);
  const permitted = `Readable **documentation** with [safe links](https://example.com) and \`code\`.`;
  assert.ok(permitted.length < MAX_INLINE_LENGTH);
  assert.equal(renderInline(permitted), 'Readable <strong>documentation</strong> with <a href="https://example.com">safe links</a> and <code>code</code>.');
});
