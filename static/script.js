// Krushi Rakshak — frontend logic
// Talks to the Flask backend routes defined in app.py:
//   POST /api/detect     -> analyse an uploaded photo
//   GET  /api/history    -> recent reports
//   GET  /api/hotspots   -> repeated-disease locations
//   POST /api/feedback   -> confirm/correct a diagnosis

const form = document.getElementById("detect-form");
const imageInput = document.getElementById("image-input");
const dropLabel = document.getElementById("file-drop-label");
const preview = document.getElementById("preview");
const submitBtn = document.getElementById("submit-btn");
const resultBox = document.getElementById("result");
const errorMsg = document.getElementById("error-msg");
let lastReportId = null;

// Show a preview of the chosen photo
imageInput.addEventListener("change", () => {
  const file = imageInput.files[0];
  if (!file) return;
  dropLabel.textContent = file.name;
  preview.src = URL.createObjectURL(file);
  preview.hidden = false;
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  errorMsg.hidden = true;
  resultBox.hidden = true;

  submitBtn.disabled = true;
  submitBtn.textContent = "Analysing…";

  try {
    const formData = new FormData(form);
    const response = await fetch("/api/detect", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Something went wrong.");
    }

    showResult(data);
    lastReportId = data.report_id;
    document.getElementById("fb-thanks").hidden = true;
    loadHistory();
    loadHotspots();
  } catch (err) {
    errorMsg.textContent = err.message;
    errorMsg.hidden = false;
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Analyse photo";
  }
});

function showResult(data) {
  const badge = document.getElementById("risk-badge");
  badge.textContent = data.risk_level + " risk";
  badge.className = "badge " + data.risk_level;

  document.getElementById("result-disease").textContent = data.disease;
  document.getElementById("result-confidence").textContent =
    `Detected for ${data.crop} · confidence ${Math.round(data.confidence * 100)}%`;

  const adviceList = document.getElementById("result-advice");
  adviceList.innerHTML = "";
  data.advice.forEach((step) => {
    const li = document.createElement("li");
    li.textContent = step;
    adviceList.appendChild(li);
  });

  resultBox.hidden = false;
}

// Feedback buttons feed the "Feedback & Model Improvement" loop from the PPT
document.getElementById("fb-yes").addEventListener("click", () => sendFeedback(true));
document.getElementById("fb-no").addEventListener("click", () => sendFeedback(false));

async function sendFeedback(wasCorrect) {
  if (!lastReportId) return;
  await fetch("/api/feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ report_id: lastReportId, was_correct: wasCorrect }),
  });
  document.getElementById("fb-thanks").hidden = false;
}

async function loadHistory() {
  const rows = await (await fetch("/api/history")).json();
  const tbody = document.querySelector("#history-table tbody");
  tbody.innerHTML = "";
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${new Date(row.created_at).toLocaleString()}</td>
      <td>${row.crop || "—"}</td>
      <td>${row.disease}</td>
      <td>${row.risk_level}</td>
      <td>${row.location_name || "—"}</td>
    `;
    tbody.appendChild(tr);
  });
}

async function loadHotspots() {
  const rows = await (await fetch("/api/hotspots")).json();
  const tbody = document.querySelector("#hotspot-table tbody");
  tbody.innerHTML = "";
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${row.location_name}</td>
      <td>${row.disease}</td>
      <td>${row.report_count}</td>
      <td>${new Date(row.last_seen).toLocaleString()}</td>
    `;
    tbody.appendChild(tr);
  });
}

// Initial load
loadHistory();
loadHotspots();
