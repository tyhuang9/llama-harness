import { existsSync, lstatSync, mkdirSync, readFileSync, readdirSync, realpathSync, statSync, writeFileSync } from "node:fs";
import { dirname, isAbsolute, relative, resolve, sep } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const docsDirectory = dirname(fileURLToPath(import.meta.url));
const repositoryDirectory = resolve(docsDirectory, "..");
const realRepositoryDirectory = realpathSync(repositoryDirectory);
const repositoryUrl = "https://github.com/tyhuang9/llama-harness";
export const MAX_SOURCE_BYTES = 512 * 1024;
export const MAX_LINE_LENGTH = 16 * 1024;
export const MAX_INLINE_LENGTH = 4 * 1024;
export const MAX_INLINE_DEPTH = 16;

export const GUIDES = {
  "architecture.md": "architecture.html",
  "developer-console.md": "developer-console.html",
  "distribution.md": "distribution.html",
  "embedding.md": "embedding.html",
  "evaluations.md": "evaluations.html",
  "integrating-note.md": "integrating-note.html",
  "migration.md": "migration.html",
  "note-embedding-dependencies.md": "note-embedding-dependencies.html",
  "MILESTONES.md": "milestones.html",
  "observability.md": "observability.html",
  "promptfoo-integration.md": "promptfoo-integration.html",
  "protocol.md": "protocol.html",
  "python-sdk.md": "python-sdk.html",
  "releasing.md": "releasing.html",
  "sdk-architecture.md": "sdk-architecture.html",
  "security.md": "security.html",
  "tauri.md": "tauri.html",
  "tools-and-policies.md": "tools-and-policies.html",
  "typescript-sdk.md": "typescript-sdk.html",
};

/** Deliberately small Markdown subset used by the canonical files in docs/. */
export const SUPPORTED_MARKDOWN = Object.freeze([
  "h1-h6 headings",
  "paragraphs",
  "fenced code",
  "flat ordered and unordered lists",
  "blockquotes",
  "pipe tables",
  "inline code",
  "strong and emphasis",
  "safe links",
]);

const escapeHtml = (value) => value.replace(/[&<>\"]/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[character]);

const decodeCharacterReferences = (value) => value
  .replace(/&#(x[0-9a-f]+|\d+);?/gi, (_, code) => String.fromCodePoint(Number.parseInt(code.replace(/^x/i, ""), /^x/i.test(code) ? 16 : 10)))
  .replace(/&(colon|tab|newline);/gi, (_, name) => ({ colon: ":", tab: "\t", newline: "\n" })[name.toLowerCase()]);

export function headingSlug(value) {
  const text = value
    .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
    .replace(/[`*_~]/g, "")
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s-]/gu, "")
    .trim()
    .replace(/\s+/g, "-")
    .replace(/-+/g, "-");
  return text || "section";
}

function splitLinkTarget(target) {
  const match = target.match(/^([^?#]*)(.*)$/);
  return { path: match[1], suffix: match[2] };
}

export function resolveGuideHref(target, sourcePath) {
  const normalized = decodeCharacterReferences(target).replace(/[\u0000-\u0020\u007f-\u009f]/g, "");
  if (normalized.startsWith("//") || normalized.startsWith("\\\\")) throw new Error(`Protocol-relative or UNC link is not allowed in ${sourcePath}: ${target}`);
  if (/^[a-z]:[\\/]/i.test(normalized)) throw new Error(`Absolute local link is not allowed in ${sourcePath}: ${target}`);
  if (/^[a-z][a-z0-9+.-]*:/i.test(normalized) && !/^(https?|mailto):/i.test(normalized)) {
    throw new Error(`Unsafe link protocol in ${sourcePath}: ${target}`);
  }
  if (/^(https?|mailto):/i.test(normalized) || target.startsWith("#")) return target;

  const { path, suffix } = splitLinkTarget(target);
  if (!path) return target;
  if (isAbsolute(path)) throw new Error(`Absolute local link is not allowed in ${sourcePath}: ${target}`);
  const targetPath = resolve(docsDirectory, path);
  const unresolvedRelative = relative(repositoryDirectory, targetPath);
  if (isAbsolute(unresolvedRelative) || unresolvedRelative === ".." || unresolvedRelative.startsWith(`..${sep}`)) {
    throw new Error(`Local link escapes the repository in ${sourcePath}: ${target}`);
  }
  if (!existsSync(targetPath)) throw new Error(`Unresolved local link in ${sourcePath}: ${target}`);

  const realTargetPath = realpathSync(targetPath);
  const repositoryPath = relative(realRepositoryDirectory, realTargetPath);
  if (isAbsolute(repositoryPath) || repositoryPath === ".." || repositoryPath.startsWith(`..${sep}`)) {
    throw new Error(`Local link resolves outside the repository in ${sourcePath}: ${target}`);
  }
  const renderedGuide = Object.entries(GUIDES).find(([source]) => realpathSync(resolve(docsDirectory, source)) === realTargetPath)?.[1];
  if (renderedGuide) return `${renderedGuide}${suffix}`;
  const normalizedPath = repositoryPath.split(sep).join("/");
  if (realTargetPath.toLowerCase().endsWith(".md")) return `${repositoryUrl}/blob/main/${normalizedPath}${suffix}`;
  const kind = statSync(realTargetPath).isDirectory() ? "tree" : "blob";
  return `${repositoryUrl}/${kind}/main/${normalizedPath}${suffix}`;
}

export function renderInline(value, sourcePath = "inline.md", depth = 0) {
  if (value.length > MAX_INLINE_LENGTH) throw new Error(`Inline Markdown exceeds ${MAX_INLINE_LENGTH} characters in ${sourcePath}`);
  if (depth > MAX_INLINE_DEPTH) throw new Error(`Inline Markdown exceeds nesting depth ${MAX_INLINE_DEPTH} in ${sourcePath}`);
  const token = /`([^`]+)`|\[([^\]]+)\]\(([^\s)]+)\)|(\*\*|__)(.+?)\4|\*([^*]+)\*|_([^_]+)_/g;
  let output = "";
  let lastIndex = 0;
  for (const match of value.matchAll(token)) {
    output += escapeHtml(value.slice(lastIndex, match.index));
    if (match[1] !== undefined) output += `<code>${escapeHtml(match[1])}</code>`;
    else if (match[2] !== undefined) output += `<a href="${escapeHtml(resolveGuideHref(match[3], sourcePath))}">${renderInline(match[2], sourcePath, depth + 1)}</a>`;
    else if (match[5] !== undefined) output += `<strong>${renderInline(match[5], sourcePath, depth + 1)}</strong>`;
    else output += `<em>${renderInline(match[6] ?? match[7], sourcePath, depth + 1)}</em>`;
    lastIndex = match.index + match[0].length;
  }
  return output + escapeHtml(value.slice(lastIndex));
}

const isTableDivider = (line) => /^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*$/.test(line);
const tableCells = (line) => line.trim().replace(/^\||\|$/g, "").split("|").map((cell) => cell.trim());

export function renderMarkdown(markdown, sourcePath = "inline.md") {
  const sourceBytes = Buffer.byteLength(markdown, "utf8");
  if (sourceBytes > MAX_SOURCE_BYTES) throw new Error(`Markdown source exceeds ${MAX_SOURCE_BYTES} bytes in ${sourcePath}`);
  const lines = markdown.replace(/\r\n/g, "\n").split("\n");
  const oversizedLine = lines.find((line) => line.length > MAX_LINE_LENGTH);
  if (oversizedLine !== undefined) throw new Error(`Markdown line exceeds ${MAX_LINE_LENGTH} characters in ${sourcePath}`);
  const output = [];
  const headingCounts = new Map();
  let index = 0;
  let lastIndex = -1;

  while (index < lines.length) {
    if (index <= lastIndex) throw new Error(`Markdown parser made no progress in ${sourcePath}`);
    lastIndex = index;
    const line = lines[index];
    if (!line.trim()) { index += 1; continue; }
    if (/^\s+(?:[-*+]|\d+\.)\s+/.test(line)) throw new Error(`Nested or indented lists are not supported in ${sourcePath}: ${line.trim()}`);

    const fence = line.match(/^```\s*([^\s]*)\s*$/);
    if (fence) {
      const language = fence[1].replace(/[^a-z0-9_-]/gi, "").toLowerCase();
      const code = [];
      index += 1;
      while (index < lines.length && !/^```\s*$/.test(lines[index])) code.push(lines[index++]);
      if (index === lines.length) throw new Error(`Unclosed code fence in ${sourcePath}`);
      index += 1;
      output.push(`<pre><code${language ? ` class="language-${language}"` : ""}>${escapeHtml(code.join("\n"))}</code></pre>`);
      continue;
    }

    const heading = line.match(/^(#{1,6})\s+(.+?)\s*#*\s*$/);
    if (heading) {
      const level = heading[1].length;
      const baseSlug = headingSlug(heading[2]);
      const count = headingCounts.get(baseSlug) ?? 0;
      headingCounts.set(baseSlug, count + 1);
      const slug = count ? `${baseSlug}-${count}` : baseSlug;
      output.push(`<h${level} id="${escapeHtml(slug)}">${renderInline(heading[2], sourcePath)}</h${level}>`);
      index += 1;
      continue;
    }

    if (index + 1 < lines.length && line.includes("|") && isTableDivider(lines[index + 1])) {
      const headings = tableCells(line);
      index += 2;
      const rows = [];
      while (index < lines.length && lines[index].includes("|") && lines[index].trim()) rows.push(tableCells(lines[index++]));
      output.push(`<div class="guide-table-wrap" tabindex="0" role="region" aria-label="Documentation table"><table><thead><tr>${headings.map((cell) => `<th>${renderInline(cell, sourcePath)}</th>`).join("")}</tr></thead><tbody>${rows.map((row) => `<tr>${row.map((cell) => `<td>${renderInline(cell, sourcePath)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`);
      continue;
    }

    const quote = line.match(/^>\s?(.*)$/);
    if (quote) {
      const quoted = [];
      while (index < lines.length && (lines[index].match(/^>\s?(.*)$/))) quoted.push(lines[index++].replace(/^>\s?/, ""));
      output.push(`<blockquote><p>${renderInline(quoted.join(" "), sourcePath)}</p></blockquote>`);
      continue;
    }

    const list = line.match(/^(?:[-*+]\s+|\d+\.\s+)(.*)$/);
    if (list) {
      const ordered = /^\d+\.\s+/.test(line);
      const items = [];
      const pattern = ordered ? /^\d+\.\s+(.*)$/ : /^[-*+]\s+(.*)$/;
      while (index < lines.length) {
        const item = lines[index].match(pattern);
        if (!item) break;
        index += 1;
        const itemLines = [item[1]];
        while (index < lines.length && lines[index].trim() && !pattern.test(lines[index])) {
          if (/^\s+(?:[-*+]|\d+\.)\s+/.test(lines[index])) throw new Error(`Nested or indented lists are not supported in ${sourcePath}: ${lines[index].trim()}`);
          itemLines.push(lines[index++].trim());
        }
        items.push(`<li>${renderInline(itemLines.join(" "), sourcePath)}</li>`);
      }
      output.push(`<${ordered ? "ol" : "ul"}>${items.join("")}</${ordered ? "ol" : "ul"}>`);
      continue;
    }

    const paragraph = [];
    let paragraphLength = 0;
    while (index < lines.length && lines[index].trim() && !/^```/.test(lines[index]) && !/^(#{1,6})\s+/.test(lines[index]) && !/^>\s?/.test(lines[index]) && !/^(?:[-*+]\s+|\d+\.\s+)/.test(lines[index]) && !(index + 1 < lines.length && lines[index].includes("|") && isTableDivider(lines[index + 1]))) {
      const part = lines[index++].trim();
      paragraphLength += part.length + (paragraph.length ? 1 : 0);
      if (paragraphLength > MAX_INLINE_LENGTH) throw new Error(`Inline Markdown exceeds ${MAX_INLINE_LENGTH} characters in ${sourcePath}`);
      paragraph.push(part);
    }
    output.push(`<p>${renderInline(paragraph.join(" "), sourcePath)}</p>`);
  }
  return output.join("\n      ");
}

export function renderGuide(sourceName) {
  const markdown = readFileSync(resolve(docsDirectory, sourceName), "utf8");
  const title = markdown.match(/^#\s+(.+)$/m)?.[1] ?? sourceName;
  const body = renderMarkdown(markdown, sourceName);
  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8" /><meta name="viewport" content="width=device-width, initial-scale=1" /><meta name="description" content="${escapeHtml(title)} guide for llama-harness." /><meta name="theme-color" content="#36141c" /><title>${escapeHtml(title)} | llama-harness</title><link rel="icon" type="image/png" href="../assets/favicon.png" /><link rel="stylesheet" href="../styles.css" /></head>
<body class="guide-page"><a class="skip-link" href="#guide-content">Skip to guide content</a><header class="topbar guide-header"><a class="brand" href="../index.html" aria-label="llama-harness documentation home"><span class="brand-mark" aria-hidden="true"><img src="../assets/favicon.png" alt="" /></span><span>llama-harness</span></a><a class="guide-back-link" href="../index.html">Back to docs</a></header><main id="guide-content" class="guide-main" tabindex="-1"><article class="guide-article"><p class="guide-source">Guide source: <code>${escapeHtml(sourceName)}</code></p>${body}</article></main><footer class="guide-footer"><a href="../index.html">Back to documentation</a></footer></body>
</html>\n`;
}

export function checkOrWrite(mode) {
  if (mode !== "--check" && mode !== "--write") throw new Error("Use --check or --write");
  const failures = [];
  const guidesDirectory = resolve(docsDirectory, "guides");
  if (mode === "--write") mkdirSync(guidesDirectory, { recursive: true });
  else if (existsSync(guidesDirectory)) {
    const expectedFiles = new Set(Object.values(GUIDES));
    for (const outputName of readdirSync(guidesDirectory)) if (!expectedFiles.has(outputName)) failures.push(`unexpected ${outputName}`);
  }
  for (const [sourceName, outputName] of Object.entries(GUIDES)) {
    const outputPath = resolve(guidesDirectory, outputName);
    const expected = renderGuide(sourceName);
    if (mode === "--write") writeFileSync(outputPath, expected);
    else if (!existsSync(outputPath) || !lstatSync(outputPath).isFile() || lstatSync(outputPath).isSymbolicLink() || readFileSync(outputPath, "utf8").replace(/\r\n/g, "\n").replace(/\n+$/, "\n") !== expected) failures.push(outputName);
  }
  if (failures.length) throw new Error(`Generated guides are missing or stale: ${failures.join(", ")}. Run npm run build in docs.`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  try {
    checkOrWrite(process.argv[2]);
  } catch (error) {
    console.error(error.message);
    process.exitCode = 1;
  }
}
