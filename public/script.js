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