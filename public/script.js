/*
document.getElementById("checkBtn").addEventListener("click", () => {
  const text = document.getElementById("inputText").value;
  const resultDiv = document.getElementById("result");

  if (!text.trim()) {
    resultDiv.textContent = "Please enter some text.";
    return;
  }

  
  const isAI = text.length % 2 === 0; 
  resultDiv.textContent = isAI
    ?  "This text looks AI-generated."
    : " This text seems human-written.";
});

*/

const btn = document.getElementById("checkBtn");
const result = document.getElementById("result");
const loader = document.getElementById("loader");
const confidenceBox = document.getElementById("confidenceBox");
const confidenceFill = document.getElementById("confidenceFill");
const confidenceValue = document.getElementById("confidenceValue");
const themeToggle = document.getElementById("themeToggle");

btn.addEventListener("click", () => {
  const text = document.getElementById("inputText").value;

  result.style.display = "none";
  confidenceBox.style.display = "none";

  if (!text.trim()) {
    result.style.display = "block";
    result.className = "result ai";
    result.innerText = "⚠️ Please paste some text first.";
    return;
  }

  // Show loader
  loader.style.display = "block";

  // Fake analysis delay (backend later)
  setTimeout(() => {
    loader.style.display = "none";

    const confidence = Math.floor(Math.random() * 20) + 80; // 80–99%

    result.style.display = "block";
    result.className = "result human";
    result.innerText = "🧠 Likely Human-written";

    confidenceBox.style.display = "block";
    confidenceFill.style.width = confidence + "%";
    confidenceValue.innerText = confidence + "%";
  }, 1500);
});

/* Theme toggle */
themeToggle.addEventListener("click", () => {
  document.body.classList.toggle("light");
  themeToggle.innerText =
    document.body.classList.contains("light") ? "☀️" : "🌙";
});
