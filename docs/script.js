const menu = document.querySelector('.menu-button');
const sidebar = document.querySelector('.sidebar');
menu?.addEventListener('click', () => {
  const open = sidebar.classList.toggle('open');
  menu.setAttribute('aria-expanded', String(open));
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
