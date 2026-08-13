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
  it("maps every in-page sidebar item to an existing section and guide items to Markdown", () => {
    const { document } = loadDocs();
    const links = [...document.querySelectorAll(".side-nav a")];
    const inPage = links.filter((link) => link.getAttribute("href").startsWith("#"));

    expect(docsHtml).not.toMatch(/<a[^>]*class="[^"]*\bactive\b/);
    expect(inPage.map((link) => link.getAttribute("href"))).toEqual(["#overview", "#quickstart", "#api", "#tools", "#runs", "#integrations"]);
    inPage.forEach((link) => expect(document.querySelector(link.getAttribute("href"))).not.toBeNull());
    expect(document.querySelector('.side-nav a[href="#api"]')?.textContent).toBe("AgentRunner");
    expect(document.querySelector('.side-nav a[href="#integrations"]')?.textContent).toBe("Use the same engine anywhere");
    expect(links.filter((link) => /SDK|Tauri integration/.test(link.textContent)).map((link) => link.getAttribute("href")))
      .toEqual(["typescript-sdk.md", "python-sdk.md", "tauri.md"]);
    expect(activeLinks(document)).toHaveLength(1);
    expect(activeLinks(document)[0].getAttribute("aria-current")).toBe("location");
  });

  it("uses the hash and 80px reading threshold to maintain exactly one active entry", () => {
    const window = loadDocs("#api");
    const { document } = window;
    const positions = { overview: -300, quickstart: -160, api: -20, tools: 81, runs: 500, integrations: 700 };
    Object.entries(positions).forEach(([id, top]) => {
      document.getElementById(id).getBoundingClientRect = () => ({ top });
    });

    window.dispatchEvent(new window.Event("scroll"));
    expect(activeLinks(document)).toHaveLength(1);
    expect(activeLinks(document)[0].getAttribute("href")).toBe("#api");

    document.getElementById("tools").getBoundingClientRect = () => ({ top: 80 });
    window.dispatchEvent(new window.Event("scroll"));
    expect(activeLinks(document)).toHaveLength(1);
    expect(activeLinks(document)[0].getAttribute("href")).toBe("#tools");
    expect(activeLinks(document)[0].getAttribute("aria-current")).toBe("location");

    document.getElementById("runs").getBoundingClientRect = () => ({ top: -20 });
    document.getElementById("integrations").getBoundingClientRect = () => ({ top: 81 });
    window.dispatchEvent(new window.Event("scroll"));
    expect(activeLinks(document)[0].getAttribute("href")).toBe("#runs");

    document.getElementById("integrations").getBoundingClientRect = () => ({ top: 80 });
    window.dispatchEvent(new window.Event("scroll"));
    expect(activeLinks(document)).toHaveLength(1);
    expect(activeLinks(document)[0].getAttribute("href")).toBe("#integrations");
    expect(activeLinks(document)[0].getAttribute("aria-current")).toBe("location");
    expect(document.querySelectorAll('.side-nav a:not([href^="#"])[aria-current]')).toHaveLength(0);
  });

  it("opens and closes the mobile menu with synchronized state and link focus return", () => {
    const window = loadDocs();
    const { document } = window;
    const menu = document.querySelector(".menu-button");
    const link = document.querySelector('.side-nav a[href="#api"]');

    expect(docsStyles).toMatch(/@media\(max-width:900px\).*?\.sidebar\{[^}]*visibility:hidden/);
    expect(docsStyles).toMatch(/\.sidebar\.open\{[^}]*visibility:visible/);
    menu.click();
    expect(document.querySelector(".sidebar").classList.contains("open")).toBe(true);
    expect(menu.getAttribute("aria-expanded")).toBe("true");

    link.focus();
    link.dispatchEvent(new window.MouseEvent("click", { bubbles: true, cancelable: true }));
    expect(document.querySelector(".sidebar").classList.contains("open")).toBe(false);
    expect(menu.getAttribute("aria-expanded")).toBe("false");
    expect(document.activeElement).toBe(menu);

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
    search.value = "security";
    search.dispatchEvent(new window.Event("input", { bubbles: true }));

    expect(document.querySelector('.side-nav a[href="security.md"]').hidden).toBe(false);
    expect(document.querySelector('.side-nav a[href="#api"]').hidden).toBe(true);
  });
});
