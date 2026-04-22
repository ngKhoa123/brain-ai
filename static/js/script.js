const input = document.getElementById("fileInput");
const statusText = document.getElementById("fileStatus");
const uploadBox = document.getElementById("uploadBox");
const uploadText = document.getElementById("uploadText");
const form = document.getElementById("uploadForm");
const btn = document.getElementById("analyzeBtn");

input.addEventListener("change", () => {
    const file = input.files[0];

    if (file) {
        statusText.innerText = "✅ Selected: " + file.name;
        uploadBox.classList.add("active");
        uploadText.innerText = "📂 Change image";
    }
});

form.addEventListener("submit", () => {
    btn.innerText = "Analyzing...";
    btn.classList.add("loading");
});