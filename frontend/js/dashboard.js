// dashboard.js - main client logic for the dashboard page
import { supabase } from "./supabase.js";
import { fetchPrediction } from "./api.js";

// Elements
const sidebarLinks = document.querySelectorAll(".nav-link");
const sections = {
  dashboard: document.getElementById("dashboard-section"),
  about: document.getElementById("about-section"),
};
const emptyState = document.getElementById("empty-state");
const predictionFormContainer = document.getElementById("prediction-form");
const resultCards = document.getElementById("result-cards");
const predictForm = document.getElementById("predict-form");
const predictButton = document.getElementById("predict-button");
const predictLoading = document.getElementById("predict-loading");

// Result elements
const predictedCasesEl = document.getElementById("predicted-cases");
const riskLevelEl = document.getElementById("risk-level");
const locationInfoEl = document.getElementById("location-info");
const dataQualityEl = document.getElementById("data-quality");
const summaryMessageEl = document.getElementById("summary-message");

// Logout
const logoutLink = document.getElementById("logout-link");
logoutLink.addEventListener("click", async (e) => {
  e.preventDefault();
  await supabase.auth.signOut();
  window.location.href = "index.html";
});

// Navigation handling
function activateSection(sectionKey) {
  // hide all sections
  Object.values(sections).forEach((sec) => sec.style.display = "none");
  // show selected
  const target = sections[sectionKey];
  if (target) target.style.display = "block";
  // update active link styling
  sidebarLinks.forEach((link) => {
    link.classList.toggle("active", link.dataset.section === sectionKey);
  });
}

sidebarLinks.forEach((link) => {
  link.addEventListener("click", (e) => {
    e.preventDefault();
    const section = link.dataset.section;
    if (section) activateSection(section);
  });
});

// Session protection – redirect to login if not authenticated
async function protectPage() {
  const { data: { session } } = await supabase.auth.getSession();
  if (!session) {
    window.location.href = "index.html";
    return false;
  }
  return true;
}

// Show loading spinner for prediction request
function setPredictLoading(isLoading) {
  if (isLoading) {
    predictButton.disabled = true;
    predictLoading.classList.remove("hidden");
    predictButton.textContent = "Analyzing...";
  } else {
    predictButton.disabled = false;
    predictLoading.classList.add("hidden");
    predictButton.textContent = "Predict Dengue Activity";
  }
}

function showErrorCard(message) {
  // Hide form and result cards, show empty state with error text
  predictionFormContainer.style.display = "none";
  resultCards.style.display = "none";
  emptyState.innerHTML = `<div class="message error">${message}</div>`;
  emptyState.style.display = "block";
}

function showResult(data) {
  // Populate cards
  const pred = data.prediction;
  const loc = data.location;

  predictedCasesEl.textContent = pred.predicted_cases.toFixed(2);

  const risk = pred.risk_level.toLowerCase();
  riskLevelEl.textContent = pred.risk_level;
  riskLevelEl.className = ""; // reset
  riskLevelEl.classList.add(risk.replace(/\s+/g, "-")); // e.g., "very-high"

  locationInfoEl.innerHTML = `
    <strong>District:</strong> ${loc.district}<br/>
    <strong>State:</strong> ${loc.state}<br/>
    <strong>Latitude:</strong> ${loc.latitude}<br/>
    <strong>Longitude:</strong> ${loc.longitude}
  `;

  const dq = data.data_quality?.district_lag_features_available;
  dataQualityEl.textContent = dq ? "Data sufficient for prediction" : "Insufficient historical data for reliable prediction";

  summaryMessageEl.textContent = data.message || "";

  // UI updates
  emptyState.style.display = "none";
  predictionFormContainer.style.display = "block";
  resultCards.style.display = "grid";
}

// Form submission
predictForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const location = document.getElementById("location-input").value.trim();
  const date = document.getElementById("date-input").value;
  if (!location || !date) {
    showErrorCard("Please provide both location and date.");
    return;
  }
  setPredictLoading(true);
  try {
    const response = await fetchPrediction(location, date);
    // Basic validation of expected structure
    if (response.status !== "success" || !response.prediction_available) {
      const msg = response.message || "Prediction unavailable for the given request.";
      showErrorCard(msg);
      return;
    }
    showResult(response);
  } catch (err) {
    showErrorCard(err.message);
  } finally {
    setPredictLoading(false);
  }
});

// Initial page setup
(async () => {
  const ok = await protectPage();
  if (!ok) return;
  // default to dashboard view
  activateSection("dashboard");
  // show form (but hide result cards until a prediction is made)
  predictionFormContainer.style.display = "block";
  resultCards.style.display = "none";
  emptyState.style.display = "block";
})();
