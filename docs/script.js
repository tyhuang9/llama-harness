const menu = document.querySelector('.menu-button');
const sidebar = document.querySelector('.sidebar');

function closeSidebar() {
  if (!sidebar.classList.contains('open')) return;
  sidebar.classList.remove('open');
  menu?.setAttribute('aria-expanded', 'false');
  menu?.focus({ preventScroll: true });
}

menu?.addEventListener('click', () => {
  const open = sidebar.classList.toggle('open');
  menu.setAttribute('aria-expanded', String(open));
});

document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape') closeSidebar();
});

document.querySelectorAll('.copy').forEach((button) => {
  button.addEventListener('click', async () => {
    const value = button.dataset.copy;
    if (!value) return;
    await navigator.clipboard.writeText(value);
    const original = button.textContent;
    button.textContent = 'Copied';
    window.setTimeout(() => { button.textContent = original; }, 1400);
  });
});

const search = document.querySelector('#doc-search');
search?.addEventListener('input', () => {
  const query = search.value.trim().toLowerCase();
  document.querySelectorAll('.side-nav a').forEach((link) => {
    link.hidden = Boolean(query) && !link.textContent.toLowerCase().includes(query);
  });
});

const inPageLinks = [...document.querySelectorAll('.side-nav a[href^="#"]')];
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
  closeSidebar();
}));
