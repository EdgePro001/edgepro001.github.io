const root = document.documentElement;
let savedTheme = null;

try {
  savedTheme = window.localStorage.getItem("theme");
} catch {
  savedTheme = null;
}

if (savedTheme === "dark") {
  root.dataset.theme = "dark";
} else if (savedTheme === "light") {
  root.dataset.theme = "light";
}

document.querySelector(".theme-toggle").addEventListener("click", () => {
  const nextTheme = root.dataset.theme === "dark" ? "light" : "dark";
  root.dataset.theme = nextTheme;
  try {
    window.localStorage.setItem("theme", nextTheme);
  } catch {}
});

const notes = [
  "Write things down.",
  "Taste is learnable.",
  "Small tools matter.",
  "Ask the next question.",
  "Reason carefully.",
];

function showNote(x, y) {
  const note = document.createElement("span");
  note.className = "float-note";
  note.textContent = notes[Math.floor(Math.random() * notes.length)];
  note.style.left = `${x}px`;
  note.style.top = `${y}px`;
  document.body.append(note);
  window.setTimeout(() => note.remove(), 2400);
}

const clickNote = document.querySelector(".click-note");
if (clickNote) {
  clickNote.addEventListener("click", () => clickNote.classList.add("hidden"));
}

document.addEventListener("click", (event) => {
  if (event.target.closest("a, button")) return;
  if (window.innerWidth < 769) return;
  if (Math.random() > 0.45) return;
  showNote(event.clientX, event.clientY);
});
