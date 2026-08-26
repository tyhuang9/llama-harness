const menu = document.querySelector('.menu-button');
const sidebar = document.querySelector('.sidebar');
const search = document.querySelector('#doc-search');
const filterStatus = document.querySelector('#filter-status');
const sideLinks = [...document.querySelectorAll('.side-nav a')];
const navSections = [...document.querySelectorAll('.nav-section')];
const copyStatus = document.createElement('span');

copyStatus.className = 'sr-only';
copyStatus.setAttribute('aria-live', 'polite');
document.body.append(copyStatus);

function closeSidebar({ restoreFocus = true } = {}) {
  if (!sidebar.classList.contains('open')) return;
  sidebar.classList.remove('open');
  menu?.setAttribute('aria-expanded', 'false');
  if (restoreFocus) menu?.focus({ preventScroll: true });
}

menu?.addEventListener('click', () => {
  if (sidebar.classList.contains('open')) {
    closeSidebar({ restoreFocus: false });
    return;
  }
  sidebar.classList.add('open');
  menu.setAttribute('aria-expanded', 'true');
  sideLinks[0]?.focus({ preventScroll: true });
});

document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape') closeSidebar();
  if (event.key === '/' && !event.ctrlKey && !event.metaKey && !event.altKey) {
    const editable = event.target instanceof HTMLInputElement
      || event.target instanceof HTMLTextAreaElement
      || event.target?.isContentEditable;
    if (!editable) {
      event.preventDefault();
      search?.focus();
    }
  }
});

window.addEventListener('resize', () => {
  if (window.matchMedia('(min-width: 901px)').matches) closeSidebar({ restoreFocus: false });
});

document.querySelectorAll('.copy').forEach((button) => {
  button.addEventListener('click', async () => {
    const value = button.dataset.copy;
    if (!value) return;
    const original = button.textContent;
    button.disabled = true;
    try {
      if (!navigator.clipboard?.writeText) throw new Error('Clipboard access is unavailable');
      await navigator.clipboard.writeText(value);
      button.textContent = 'Copied';
      copyStatus.textContent = 'Copied to clipboard.';
    } catch {
      button.textContent = 'Copy failed';
      copyStatus.textContent = 'Unable to copy. Select the command or code manually.';
    }
    window.setTimeout(() => {
      button.textContent = original;
      copyStatus.textContent = '';
      button.disabled = false;
    }, 1800);
  });
});

search?.addEventListener('input', () => {
  const query = search.value.trim().toLowerCase();
  let visible = 0;
  sideLinks.forEach((link) => {
    link.hidden = Boolean(query) && !link.textContent.toLowerCase().includes(query);
    if (!link.hidden) visible += 1;
  });
  navSections.forEach((section) => {
    section.hidden = ![...section.querySelectorAll('a')].some((link) => !link.hidden);
  });
  if (filterStatus) filterStatus.textContent = query
    ? `${visible} matching ${visible === 1 ? 'link' : 'links'}.`
    : '';
});

const inPageLinks = [...document.querySelectorAll('.side-nav a[href^="#"], .toc-nav a[href^="#"]')];
const sections = inPageLinks
  .map((link) => ({ link, section: document.querySelector(link.getAttribute('href')) }))
  .filter(({ section }) => section);

function setActiveLink(hash) {
  let match;
  inPageLinks.forEach((link) => {
    const active = link.getAttribute('href') === hash;
    link.classList.toggle('active', active);
    if (active) link.setAttribute('aria-current', 'location');
    else link.removeAttribute('aria-current');
    if (active) match = link;
  });
  return match;
}

function updateActiveLink() {
  const current = sections.reduce((active, item) => (
    item.section.getBoundingClientRect().top <= 80 ? item : active
  ), sections[0]);
  if (current) setActiveLink(current.link.getAttribute('href'));
}

if (!setActiveLink(window.location.hash)) updateActiveLink();
window.addEventListener('hashchange', () => setActiveLink(window.location.hash) || updateActiveLink());
window.addEventListener('scroll', updateActiveLink, { passive: true });

inPageLinks.forEach((link) => link.addEventListener('click', () => {
  setActiveLink(link.getAttribute('href'));
  closeSidebar({ restoreFocus: false });
}));
