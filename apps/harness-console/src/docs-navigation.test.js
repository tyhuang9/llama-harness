import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { JSDOM } from "jsdom";
import { afterEach, describe, expect, it } from "vitest";

const docsHtml = readFileSync(resolve(process.cwd(), "../../docs/index.html"), "utf8");
const docsScript = readFileSync(resolve(process.cwd(), "../../docs/script.js"), "utf8");
const docsStyles = readFileSync(resolve(process.cwd(), "../../docs/styles.css"), "utf8");
let dom;

function loadDocs(hash = "") {
  dom = new JSDOM(docsHtml, {
    runScripts: "outside-only",
    url: `https://example.test/docs/index.html${hash}`,
  });
  dom.window.requestAnimationFrame = (callback) => {
    callback();
    return 0;
  };
  dom.window.eval(docsScript);
  return dom.window;
}

function activeLinks(document) {
  return [...document.querySelectorAll(".side-nav a.active")];
}

afterEach(() => {
  dom?.window.close();
  dom = undefined;
});

describe("documentation navigation", () => {
  it("maps every sidebar item to an existing in-page section", () => {
    const { document } = loadDocs("#overview");
    const links = [...document.querySelectorAll('.side-nav a[href^="#"]')];
    const expected = new Map([
      ["#overview", "Introduction"],
      ["#installation", "Installation"],
      ["#quickstart", "Quick start"],
      ["#ownership", "Application ownership"],
      ["#concepts", "Core concepts"],
      ["#features", "Cargo features"],
    ]);

    expect(docsHtml).not.toMatch(/<a[^>]*class="[^"]*\bactive\b/);
    expect(links.map((link) => link.getAttribute("href"))).toEqual([...expected.keys()]);
    links.forEach((link) => {
      const href = link.getAttribute("href");
      expect(href).toMatch(/^#/);
      const target = document.querySelector(href);
      expect(target).not.toBeNull();
      expect(link.textContent.trim()).toBe(expected.get(href));
      expect(target.querySelector("h1, h2")).not.toBeNull();
    });
    ["embedding", "tools-and-policies", "observability", "evaluations", "tauri", "security", "architecture"].forEach((guide) => {
      expect(document.querySelector(`.side-nav a[href="guides/${guide}.html"]`)).not.toBeNull();
    });
    expect(activeLinks(document)).toHaveLength(1);
    expect(activeLinks(document)[0].getAttribute("aria-current")).toBe("location");
  });

  it("uses the hash and 80px reading threshold to maintain exactly one active entry", () => {
    const window = loadDocs("#quickstart");
    const { document } = window;
    const positions = { overview: -300, installation: -160, quickstart: -20, ownership: 81, concepts: 500, features: 700, guides: 900 };
    Object.entries(positions).forEach(([id, top]) => {
      document.getElementById(id).getBoundingClientRect = () => ({ top });
    });

    window.dispatchEvent(new window.Event("scroll"));
    expect(activeLinks(document)).toHaveLength(1);
    expect(activeLinks(document)[0].getAttribute("href")).toBe("#quickstart");

    document.getElementById("ownership").getBoundingClientRect = () => ({ top: 80 });
    window.dispatchEvent(new window.Event("scroll"));
    expect(activeLinks(document)).toHaveLength(1);
    expect(activeLinks(document)[0].getAttribute("href")).toBe("#ownership");
    expect(activeLinks(document)[0].getAttribute("aria-current")).toBe("location");

    document.getElementById("concepts").getBoundingClientRect = () => ({ top: -20 });
    document.getElementById("features").getBoundingClientRect = () => ({ top: 81 });
    window.dispatchEvent(new window.Event("scroll"));
    expect(activeLinks(document)[0].getAttribute("href")).toBe("#concepts");

    document.getElementById("features").getBoundingClientRect = () => ({ top: 80 });
    window.dispatchEvent(new window.Event("scroll"));
    expect(activeLinks(document)).toHaveLength(1);
    expect(activeLinks(document)[0].getAttribute("href")).toBe("#features");
    expect(activeLinks(document)[0].getAttribute("aria-current")).toBe("location");
    expect(document.querySelectorAll('.side-nav a:not([href^="#"])[aria-current]')).toHaveLength(0);
  });

  it("opens and closes the mobile menu with synchronized state and link focus return", () => {
    const window = loadDocs();
    const { document } = window;
    const menu = document.querySelector(".menu-button");
    const link = document.querySelector('.side-nav a[href="#quickstart"]');

    expect(docsStyles).toMatch(/@media\s*\(max-width:\s*800px\)[\s\S]*?\.sidebar\s*\{[^}]*visibility:\s*hidden/);
    expect(docsStyles).toMatch(/\.sidebar\.open\s*\{[^}]*visibility:\s*visible/);
    menu.click();
    expect(document.querySelector(".sidebar").classList.contains("open")).toBe(true);
    expect(menu.getAttribute("aria-expanded")).toBe("true");

    link.focus();
    link.dispatchEvent(new window.MouseEvent("click", { bubbles: true, cancelable: true }));
    expect(document.querySelector(".sidebar").classList.contains("open")).toBe(false);
    expect(menu.getAttribute("aria-expanded")).toBe("false");
    expect(document.activeElement).toBe(document.getElementById("quickstart"));

    link.focus();
    link.dispatchEvent(new window.MouseEvent("click", { bubbles: true, cancelable: true }));
    expect(document.activeElement).toBe(link);
  });

  it("closes the mobile menu on Escape and returns focus only when it was open", () => {
    const window = loadDocs();
    const { document } = window;
    const menu = document.querySelector(".menu-button");
    const search = document.getElementById("doc-search");

    menu.click();
    search.focus();
    document.dispatchEvent(new window.KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
    expect(document.querySelector(".sidebar").classList.contains("open")).toBe(false);
    expect(menu.getAttribute("aria-expanded")).toBe("false");
    expect(document.activeElement).toBe(menu);

    search.focus();
    document.dispatchEvent(new window.KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
    expect(document.activeElement).toBe(search);
  });

  it("preserves sidebar search filtering", () => {
    const window = loadDocs();
    const { document } = window;
    const search = document.getElementById("doc-search");
    search.value = "observ";
    search.dispatchEvent(new window.Event("input", { bubbles: true }));

    expect(document.querySelector('.side-nav a[href="guides/observability.html"]').hidden).toBe(false);
    expect(document.querySelector('.side-nav a[href="guides/embedding.html"]').hidden).toBe(true);
  });
});
