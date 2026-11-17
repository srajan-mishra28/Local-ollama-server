// Apply the saved theme before first paint, to avoid a flash of the wrong one.
try {
  var saved = localStorage.getItem('theme');
  if (saved) document.documentElement.dataset.theme = saved;
} catch (e) { /* storage blocked */ }
