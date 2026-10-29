// dashboard.js - Main client logic and view routing for the dashboard
import { supabase } from "./supabase.js";
import {
  fetchPrediction,
  fetchSurveillanceMetrics,
  fetchXaiExplanation,
  fetchXaiImportance,
  getDistrictHeatmapPoints,
} from "./api.js";

// ─── Section Definitions ──────────────────────────────────────────────────────
const sections = {
  overview:   document.getElementById("overview-section"),
  heatmap:    document.getElementById("heatmap-section"),
  xai:        document.getElementById("xai-section"),
  prediction: document.getElementById("prediction-section"),
  about:      document.getElementById("about-section"),
};

const sectionMeta = {
  overview: {
    title: "Overview Dashboard",
    subtitle: "National and state-level epidemiological surveillance indicators",
  },
  heatmap: {
    title: "India Dengue Risk Heatmap",
    subtitle: "Latest historical district case reports from the local cleaned dataset",
  },
  xai: {
    title: "Explainable AI (XAI)",
    subtitle: "Global feature importance from the trained model, with an optional Gemini summary",
  },
  prediction: {
    title: "District Outbreak Forecasting",
    subtitle: "Generate next-report dengue activity forecasts for eligible administrative districts",
  },
  about: {
    title: "Model & Metadata Details",
    subtitle: "Technical architecture, data provenance, and public-health advisory notices",
  },
};

const pageTitle      = document.getElementById("page-title");
const pageSubtitle   = document.getElementById("page-subtitle");
const navLinks       = document.querySelectorAll(".nav-link[data-section]");
const sidebar        = document.getElementById("sidebar");
const sidebarBackdrop = document.getElementById("sidebar-backdrop");
const menuToggle     = document.getElementById("menu-toggle");

// ─── Prediction DOM ───────────────────────────────────────────────────────────
const predictForm    = document.getElementById("predict-form");
const predictButton  = document.getElementById("predict-button");
const predictBtnText = document.getElementById("predict-btn-text");
const predictLoading = document.getElementById("predict-loading");
const emptyState     = document.getElementById("empty-state");
const resultCards    = document.getElementById("result-cards");

const predictedCasesEl = document.getElementById("predicted-cases");
const riskLevelEl      = document.getElementById("risk-level");
const locationInfoEl   = document.getElementById("location-info");
const dataQualityEl    = document.getElementById("data-quality");
const summaryMessageEl = document.getElementById("summary-message");

// ─── Mobile Menu Toggle ───────────────────────────────────────────────────────
if (menuToggle && sidebar && sidebarBackdrop) {
  menuToggle.addEventListener("click", () => {
    sidebar.classList.toggle("open");
    sidebarBackdrop.classList.toggle("active");
  });

  sidebarBackdrop.addEventListener("click", () => {
    sidebar.classList.remove("open");
    sidebarBackdrop.classList.remove("active");
  });
}

// ─── Hash Routing ─────────────────────────────────────────────────────────────
/**
 * Parse section key and optional query params from the URL hash.
 * Handles both "#overview" and "#prediction?district=Pune" formats.
 */
function parseHash(hash) {
  const withoutHash = (hash || "").replace(/^#/, "");
  const [section, queryString] = withoutHash.split("?");
  const params = {};
  if (queryString) new URLSearchParams(queryString).forEach((value, key) => { params[key] = value; });
  return { section: section || "overview", params };
}

// ─── Section Activation ───────────────────────────────────────────────────────
let leafletMapInitialized = false;

function activateSection(sectionKey, params = {}) {
  if (!sections[sectionKey]) {
    sectionKey = "overview";
  }

  // Show correct section, hide all others
  Object.entries(sections).forEach(([key, element]) => {
    if (element) {
      element.classList.toggle("active", key === sectionKey);
    }
  });

  // Update active nav link
  navLinks.forEach(link => {
    link.classList.toggle("active", link.dataset.section === sectionKey);
  });

  // Update header context bar
  if (sectionMeta[sectionKey] && pageTitle && pageSubtitle) {
    pageTitle.textContent   = sectionMeta[sectionKey].title;
    pageSubtitle.textContent = sectionMeta[sectionKey].subtitle;
  }

  // Close mobile sidebar
  if (sidebar && sidebar.classList.contains("open")) {
    sidebar.classList.remove("open");
    if (sidebarBackdrop) sidebarBackdrop.classList.remove("active");
  }

  // Handle section-specific side effects
  if (sectionKey === "heatmap") {
    initLeafletMap();
  }

  if (sectionKey === "xai") {
    loadXaiImportance();
  }

  if (sectionKey === "prediction" && params.district) {
    const locationInput = document.getElementById("location-input");
    const dateInput     = document.getElementById("date-input");
    if (locationInput) locationInput.value = params.district;
    if (dateInput && !dateInput.value) {
      dateInput.value = new Date().toISOString().split("T")[0];
    }
    locationInput?.focus();
  }
}

// ─── Nav Link Click Handlers ──────────────────────────────────────────────────
navLinks.forEach(link => {
  link.addEventListener("click", e => {
    e.preventDefault();
    const section = link.dataset.section;
    if (section) {
      window.location.hash = section;
      // activateSection is called by hashchange listener
    }
  });
});

// ─── Hash Change Listener ─────────────────────────────────────────────────────
window.addEventListener("hashchange", () => {
  const { section, params } = parseHash(window.location.hash);
  activateSection(section, params);
});

// ─── Logout Handler ───────────────────────────────────────────────────────────
const logoutLink = document.getElementById("logout-link");
if (logoutLink) {
  logoutLink.addEventListener("click", async e => {
    e.preventDefault();
    localStorage.removeItem("dengue_demo_session");
    try {
      await supabase.auth.signOut();
    } catch (err) {
      console.warn("Sign-out warning:", err);
    }
    window.location.href = "index.html";
  });
}

// ─── Session Guard ────────────────────────────────────────────────────────────
async function protectPage() {
  if (localStorage.getItem("dengue_demo_session") === "true") return true;
  try {
    const { data: { session } } = await supabase.auth.getSession();
    if (!session) {
      window.location.href = "index.html";
      return false;
    }
    return true;
  } catch (err) {
    console.warn("Session verification fallback:", err);
    return true;
  }
}

// ─── Prediction Form ──────────────────────────────────────────────────────────
function setPredictLoading(isLoading) {
  if (isLoading) {
    predictButton.disabled = true;
    predictLoading.classList.remove("hidden");
    if (predictBtnText) predictBtnText.textContent = "Analyzing...";
  } else {
    predictButton.disabled = false;
    predictLoading.classList.add("hidden");
    if (predictBtnText) predictBtnText.textContent = "Generate Forecast";
  }
}

function showErrorCard(message) {
  if (resultCards) resultCards.classList.add("hidden");
  if (emptyState) {
    emptyState.innerHTML = `
      <div class="message error" style="margin: 0 auto; max-width: 500px;">
        <strong>Prediction Request Note:</strong> ${message}
      </div>
    `;
    emptyState.classList.remove("hidden");
  }
}

function showResult(data) {
  const pred = data.prediction || {};
  const loc  = data.location  || {};

  if (predictedCasesEl) {
    predictedCasesEl.textContent = typeof pred.predicted_cases === "number"
      ? pred.predicted_cases.toFixed(2)
      : (pred.predicted_cases || "N/A");
  }

  if (riskLevelEl) {
    const risk = (pred.risk_level || "Unknown").toLowerCase();
    riskLevelEl.textContent = pred.risk_level || "Assessed";
    riskLevelEl.className   = "card-value";
    riskLevelEl.classList.add(`risk-${risk.replace(/\s+/g, "-")}`);
  }

  if (locationInfoEl) {
    locationInfoEl.innerHTML = `
      <strong>District:</strong> ${loc.district || "—"}<br/>
      <strong>State:</strong> ${loc.state || "—"}<br/>
      <strong>Lat/Long:</strong> ${loc.latitude || "—"}, ${loc.longitude || "—"}
    `;
  }

  if (dataQualityEl) {
    const dq = data.data_quality?.district_lag_features_available;
    dataQualityEl.textContent = dq
      ? "Sufficient historical surveillance records"
      : "Limited historical reports for selected district";
  }

  if (summaryMessageEl) {
    summaryMessageEl.textContent = data.message || "Forecast generated based on available model features.";
  }

  if (emptyState) emptyState.classList.add("hidden");
  if (resultCards) resultCards.classList.remove("hidden");
}

if (predictForm) {
  predictForm.addEventListener("submit", async e => {
    e.preventDefault();
    const location = document.getElementById("location-input")?.value?.trim();
    const date     = document.getElementById("date-input")?.value;

    if (!location || !date) {
      showErrorCard("Please specify both a target district and an observation date.");
      return;
    }

    setPredictLoading(true);
    try {
      const response = await fetchPrediction(location, date);
      if (response.status !== "success" || !response.prediction_available) {
        showErrorCard(response.message || "Prediction currently unavailable for this location.");
        return;
      }
      showResult(response);
    } catch (err) {
      showErrorCard(err.message || "Error communicating with prediction service.");
    } finally {
      setPredictLoading(false);
    }
  });
}

// ─── Explainable AI ──────────────────────────────────────────────────────────
const xaiStatus = document.getElementById("xai-status");
const xaiImportanceList = document.getElementById("xai-importance-list");
const xaiModelNote = document.getElementById("xai-model-note");
const xaiGenerateButton = document.getElementById("xai-generate-button");
const xaiExplanationCard = document.getElementById("xai-explanation-card");
const xaiExplanation = document.getElementById("xai-explanation");
let xaiImportanceLoaded = false;
let xaiGeminiConfigured = false;

const FEATURE_LABELS = {
  state_ut: "State / UT",
  district: "District",
  current_cases: "Current reported cases",
  previous_cases: "Previous report cases",
  cases_lag_2: "Cases, 2 reports earlier",
  cases_lag_3: "Cases, 3 reports earlier",
  cases_lag_4: "Cases, 4 reports earlier",
  previous_case_change: "Change from previous report",
  previous_case_growth: "Previous case growth",
  previous_cases_rolling_mean_4: "4 report mean cases",
  previous_cases_rolling_median_4: "4 report median cases",
  previous_cases_rolling_min_4: "4 report minimum cases",
  previous_cases_rolling_max_4: "4 report maximum cases",
  previous_cases_rolling_std_4: "4 report case variation",
  days_since_previous_report: "Days since previous report",
  previous_reporting_gap_mean_4: "Mean reporting interval",
  previous_reporting_gap_median_4: "Median reporting interval",
  previous_reporting_gap_std_4: "Reporting interval variation",
  observation_year: "Observation year",
  observation_month: "Observation month",
  observation_week_of_year: "Observation week",
  observation_day_of_year: "Observation day of year",
  month_sin: "Seasonal month signal (sin)",
  month_cos: "Seasonal month signal (cos)",
  day_of_year_sin: "Annual day signal (sin)",
  day_of_year_cos: "Annual day signal (cos)",
  season: "Season",
  Latitude: "Latitude",
  Longitude: "Longitude",
  Temp: "Temperature",
  preci: "Precipitation",
  LAI: "Leaf area index (LAI)",
};

function renderXaiImportance(features) {
  if (!xaiImportanceList) return;
  xaiImportanceList.replaceChildren();
  const maximum = Math.max(...features.map(item => item.importance), 0.01);

  features.forEach(item => {
    const row = document.createElement("li");
    row.className = "xai-importance-row";

    const label = document.createElement("span");
    label.className = "xai-feature-label";
    label.textContent = FEATURE_LABELS[item.feature] || item.feature;

    const track = document.createElement("span");
    track.className = "xai-bar-track";
    track.setAttribute("aria-hidden", "true");
    const bar = document.createElement("span");
    bar.className = "xai-bar-fill";
    bar.style.width = `${Math.max(2, item.importance / maximum * 100)}%`;
    track.append(bar);

    const value = document.createElement("strong");
    value.className = "xai-feature-value";
    value.textContent = `${Number(item.importance).toFixed(2)}%`;

    row.append(label, track, value);
    xaiImportanceList.append(row);
  });
}

async function loadXaiImportance(force = false) {
  if (!xaiStatus || (xaiImportanceLoaded && !force)) return;
  xaiStatus.textContent = "Loading feature importance from the local model…";
  if (xaiGenerateButton) xaiGenerateButton.disabled = true;

  try {
    const data = await fetchXaiImportance();
    renderXaiImportance(data.features || []);
    xaiGeminiConfigured = Boolean(data.gemini_configured);
    xaiImportanceLoaded = true;
    if (xaiGenerateButton) xaiGenerateButton.disabled = !xaiGeminiConfigured;
    const configuredStatus = xaiGeminiConfigured
      ? "Gemini explanations are generated only when you request one."
      : "Set GEMINI_API_KEY for the Python API, then restart it to enable Gemini explanations.";
    xaiStatus.textContent = `Importance loaded for ${data.feature_count} of ${data.api_feature_count} API input features. ${configuredStatus}`;
    if (xaiModelNote && data.features_without_importance?.length) {
      const missing = data.features_without_importance.map(feature => FEATURE_LABELS[feature] || feature).join(", ");
      xaiModelNote.textContent = `The saved model pipeline has no feature-importance values for these API inputs: ${missing}.`;
      xaiModelNote.classList.remove("hidden");
    }
  } catch (error) {
    xaiStatus.textContent = `${error.message} Start the Python model API and reopen this page to retry.`;
  }
}

if (xaiGenerateButton) {
  xaiGenerateButton.addEventListener("click", async () => {
    if (!xaiGeminiConfigured) return;
    xaiGenerateButton.disabled = true;
    xaiGenerateButton.textContent = "Generating explanation…";
    xaiStatus.textContent = "Asking Gemini to summarize the model’s global importance results…";
    if (xaiExplanationCard) xaiExplanationCard.classList.add("hidden");

    try {
      const result = await fetchXaiExplanation();
      if (xaiExplanation) xaiExplanation.textContent = result.explanation;
      if (xaiExplanationCard) xaiExplanationCard.classList.remove("hidden");
      xaiStatus.textContent = `Explanation generated with ${result.gemini_model}.`;
    } catch (error) {
      xaiStatus.textContent = error.message;
    } finally {
      xaiGenerateButton.disabled = false;
      xaiGenerateButton.textContent = "Explain with Gemini";
    }
  });
}

// ─── KPI Counter Animation ────────────────────────────────────────────────────
function animateCounter(element, targetValue, duration = 1200) {
  if (!element || typeof targetValue !== "number") return;
  const startTime = performance.now();

  function updateCount(currentTime) {
    const elapsed  = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1);
    const easeOut  = 1 - Math.pow(1 - progress, 3);
    const currentVal = Math.round((targetValue) * easeOut);
    element.textContent = currentVal.toLocaleString("en-IN");
    if (progress < 1) {
      requestAnimationFrame(updateCount);
    } else {
      element.textContent = targetValue.toLocaleString("en-IN");
    }
  }
  requestAnimationFrame(updateCount);
}

// ─── Leaderboard Table ────────────────────────────────────────────────────────
let currentFilter    = "ALL";
let cachedMetricsData = null;

function populateLeaderboardTable(filter = "ALL") {
  const tbody = document.getElementById("leaderboard-body");
  if (!tbody || !cachedMetricsData) return;

  const districts = cachedMetricsData.top_vulnerable_districts || [];
  const filtered  = filter === "ALL"
    ? districts
    : districts.filter(d => d.risk_level === filter);

  if (filtered.length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="8" style="text-align: center; padding: 2rem; color: var(--color-text-muted);">
          No districts found for selected filter "${filter}".
        </td>
      </tr>
    `;
    return;
  }

  tbody.innerHTML = filtered.map(d => {
    const rankClass    = d.rank === 1 ? "rank-1" : (d.rank === 2 ? "rank-2" : (d.rank === 3 ? "rank-3" : ""));
    const riskBadge    = `badge-risk-${d.risk_level.toLowerCase().replace(/\s+/g, "-")}`;
    const statusHtml   = d.alert_threshold_exceeded
      ? `<span class="status-alert">
           <svg width="14" height="14" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd"/></svg>
           Alert (≥ 50)
         </span>`
      : `<span class="status-alert-normal">Normal</span>`;

    return `
      <tr>
        <td><span class="rank-badge ${rankClass}">#${d.rank}</span></td>
        <td><span class="district-name">${d.district}</span></td>
        <td><span style="color: var(--color-text-muted);">${d.state_ut}</span></td>
        <td style="text-align: right;"><span class="cases-num">${d.latest_cases.toLocaleString("en-IN")}</span></td>
        <td style="text-align: right;"><span class="cases-num" style="color: var(--color-primary);">${d.predicted_cases.toLocaleString("en-IN")}</span></td>
        <td style="text-align: center;"><span class="badge-risk ${riskBadge}">${d.risk_level}</span></td>
        <td style="text-align: center;">${statusHtml}</td>
        <td style="text-align: right;">
          <button type="button" class="btn-table-action" data-district="${d.district}">
            Forecast →
          </button>
        </td>
      </tr>
    `;
  }).join("");

  // Wire up "Forecast →" buttons to navigate to prediction page with district prefilled
  tbody.querySelectorAll(".btn-table-action").forEach(btn => {
    btn.addEventListener("click", () => {
      const districtName = btn.getAttribute("data-district");
      window.location.hash = `prediction?district=${encodeURIComponent(districtName)}`;
    });
  });
}

function initLeaderboardFilters() {
  const chips = document.querySelectorAll(".filter-chip");
  chips.forEach(chip => {
    chip.addEventListener("click", () => {
      chips.forEach(c => c.classList.remove("active"));
      chip.classList.add("active");
      currentFilter = chip.getAttribute("data-filter") || "ALL";
      populateLeaderboardTable(currentFilter);
    });
  });
}

async function loadSurveillanceKPIs() {
  const kpiHistObs  = document.getElementById("kpi-historical-obs");
  const kpiEligible = document.getElementById("kpi-eligible-districts");
  const kpiPredicted = document.getElementById("kpi-predicted-cases");
  const kpiHighRisk = document.getElementById("kpi-high-risk-districts");

  try {
    const metrics = await fetchSurveillanceMetrics();
    if (!metrics) return;
    cachedMetricsData = metrics;

    if (kpiHistObs)   animateCounter(kpiHistObs,   metrics.historical_observations || 1618);
    if (kpiEligible)  animateCounter(kpiEligible,  metrics.eligible_districts       || 202);
    if (kpiPredicted) animateCounter(kpiPredicted, metrics.predicted_cases_national_sum || 7341);
    if (kpiHighRisk)  animateCounter(kpiHighRisk,  metrics.high_risk_districts      || 15);

    populateLeaderboardTable(currentFilter);
  } catch (err) {
    console.error("Error loading surveillance KPIs:", err);
    // Display fallback static values
    if (kpiHistObs)   kpiHistObs.textContent   = "1,618";
    if (kpiEligible)  kpiEligible.textContent  = "202";
    if (kpiPredicted) kpiPredicted.textContent = "7,341";
    if (kpiHighRisk)  kpiHighRisk.textContent  = "15";
  }
}

// ─── Leaflet Heatmap ──────────────────────────────────────────────────────────
function initLeafletMap() {
  // Only initialise once; Leaflet throws if you re-init into the same container
  if (leafletMapInitialized) return;

  const mapContainer = document.getElementById("leaflet-map");
  if (!mapContainer || typeof window.L === "undefined") return;

  const map = window.L.map("leaflet-map", {
    center:  [22.5, 82.0],
    zoom:    5,
    minZoom: 4,
    maxZoom: 10,
  });

  window.L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "© <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a> contributors",
    maxZoom: 18,
  }).addTo(map);

  const points = getDistrictHeatmapPoints();
  points.forEach(pt => {
      const radius = pt.latest_cases > 200 ? 10 : pt.latest_cases > 25 ? 8 : pt.latest_cases >= 10 ? 6 : 5;
      const color = pt.latest_cases > 200 ? "#ef4444" : pt.latest_cases > 25 ? "#f97316" : pt.latest_cases >= 10 ? "#f59e0b" : "#10b981";

      const circle = window.L.circleMarker([pt.lat, pt.lng], {
        radius,
        color,
        fillColor:   color,
        fillOpacity: 0.75,
        weight:      1.5,
        opacity:     0.9,
      }).addTo(map);

      circle.bindPopup(`
        <div style="font-family: system-ui, sans-serif; min-width: 180px;">
          <div style="font-weight: 700; font-size: 0.95rem; margin-bottom: 4px;">${pt.district}</div>
          <div style="color: #64748b; font-size: 0.8rem; margin-bottom: 8px;">${pt.state_ut}</div>
          <table style="width: 100%; font-size: 0.8rem; border-collapse: collapse;">
            <tr><td style="color:#64748b; padding: 2px 0;">Latest reported cases</td><td style="text-align:right; font-weight:600;">${pt.latest_cases.toLocaleString("en-IN")}</td></tr>
            <tr><td style="color:#64748b; padding: 2px 0;">Report date</td><td style="text-align:right; font-weight:600;">${pt.last_report_date}</td></tr>
          </table>
        </div>
      `, { maxWidth: 260 });
    });
  leafletMapInitialized = true;
  requestAnimationFrame(() => map.invalidateSize());
}

// ─── Initialisation ───────────────────────────────────────────────────────────
(async () => {
  const isAuth = await protectPage();
  if (!isAuth) return;

  // Determine initial section from URL hash (handles both #overview and #prediction?district=Pune)
  const { section: initialSection, params: initialParams } = parseHash(window.location.hash);
  activateSection(initialSection, initialParams);

  // Setup leaderboard filter chips
  initLeaderboardFilters();

  // Load KPI cards and leaderboard from Python API (or fallback static data)
  loadSurveillanceKPIs();
})();
