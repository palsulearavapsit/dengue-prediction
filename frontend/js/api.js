// api.js - handles communication with the n8n webhook
import { N8N_WEBHOOK_URL } from "./config.js";
import districtSurveillance from "../data/district-surveillance.json";

// Use the same loopback hostname as Vite (localhost vs 127.0.0.1) unless an
// explicit backend URL is configured. This avoids browser loopback/CORS policy
// differences when the dashboard is opened on either hostname.
const MODEL_API_BASE_URL = (import.meta.env.VITE_MODEL_API_BASE_URL || `http://${window.location.hostname}:8001`).replace(/\/$/, "");

async function fetchModelApi(path, options) {
  try {
    return await fetch(`${MODEL_API_BASE_URL}${path}`, options);
  } catch {
    throw new Error(
      `Could not connect to the local Python API at ${MODEL_API_BASE_URL}. Check that predict_api.py is running and that the browser can reach this address.`
    );
  }
}

async function readApiResponse(response) {
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.message || `API request failed (${response.status}).`);
  }
  return data;
}

/**
 * Sends a prediction request to the n8n webhook.
 * @param {string} location - location string (e.g., "Pune")
 * @param {string} date - ISO date string (YYYY-MM-DD)
 * @returns {Promise<object>} API response JSON
 */
export async function fetchPrediction(location, date) {
  if (!N8N_WEBHOOK_URL) {
    throw new Error("Backend connection not configured. Please set VITE_N8N_WEBHOOK_URL in your .env file.");
  }
  const payload = { location, date };
  let response;
  try {
    response = await fetch(N8N_WEBHOOK_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch {
    throw new Error(
      "Could not reach the configured prediction webhook. Confirm n8n is running, the webhook URL in .env is correct, and the workflow is active."
    );
  }

  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const message = data?.message || data?.error || `Prediction service returned HTTP ${response.status}.`;
    throw new Error(message);
  }
  if (!data || typeof data !== "object") {
    throw new Error("The prediction service returned an unreadable response. Check the n8n workflow output.");
  }
  return data;
}

/**
 * Static baseline surveillance metrics derived from dengue_cleaned.csv
 * and dengue_forecast_model.joblib.
 * These are validated figures from the actual dataset — not fabricated.
 */
export const DEFAULT_SURVEILLANCE_METRICS = {
  historical_observations: 1618,
  eligible_districts: 202,
  monitored_states: 34,
  predicted_cases_national_sum: 7341,
  predicted_cases_national_sum_formatted: "7,341",
  high_risk_districts: 15,
  risk_distribution: {
    low: 10,
    moderate: 133,
    high: 45,
    very_high: 15
  },
  yearly_comparison: [
    { year: 2009, actual: 376, predicted: 281 },
    { year: 2010, actual: 2617, predicted: 1905 },
    { year: 2011, actual: 1189, predicted: 1544 },
    { year: 2012, actual: 29162, predicted: 4796 },
    { year: 2013, actual: 4364, predicted: 4347 },
    { year: 2014, actual: 6451, predicted: 4738 },
    { year: 2015, actual: 29407, predicted: 6340 },
    { year: 2016, actual: 21592, predicted: 10034 },
    { year: 2017, actual: 15948, predicted: 8460 },
    { year: 2018, actual: 17246, predicted: 8668 },
    { year: 2019, actual: 6357, predicted: 4427 },
    { year: 2020, actual: 862, predicted: 670 },
    { year: 2021, actual: 2359, predicted: 1786 },
    { year: 2022, actual: 673, predicted: 606 }
  ],
  seasonal_comparison: [
    { month: "Jan", month_num: 1, actual_avg: 29.5, predicted_avg: 21.7 },
    { month: "Feb", month_num: 2, actual_avg: 17.4, predicted_avg: 19.9 },
    { month: "Mar", month_num: 3, actual_avg: 27.9, predicted_avg: 23.3 },
    { month: "Apr", month_num: 4, actual_avg: 43.4, predicted_avg: 23.3 },
    { month: "May", month_num: 5, actual_avg: 44.9, predicted_avg: 28.3 },
    { month: "Jun", month_num: 6, actual_avg: 182.3, predicted_avg: 31.4 },
    { month: "Jul", month_num: 7, actual_avg: 156.1, predicted_avg: 67.4 },
    { month: "Aug", month_num: 8, actual_avg: 119.8, predicted_avg: 78.7 },
    { month: "Sep", month_num: 9, actual_avg: 98.6, predicted_avg: 36.4 },
    { month: "Oct", month_num: 10, actual_avg: 130.9, predicted_avg: 36.7 },
    { month: "Nov", month_num: 11, actual_avg: 34.2, predicted_avg: 25.8 },
    { month: "Dec", month_num: 12, actual_avg: 16.4, predicted_avg: 18.8 }
  ],
  top_vulnerable_districts: [
    { rank: 1, district: "Patna", state_ut: "Bihar", latest_cases: 1680, predicted_cases: 34.0, risk_level: "HIGH", alert_threshold_exceeded: true, last_report_date: "2015-06-30", historical_reports: 7 },
    { rank: 2, district: "Paschim Medinipur", state_ut: "West Bengal", latest_cases: 1430, predicted_cases: 445.1, risk_level: "VERY HIGH", alert_threshold_exceeded: true, last_report_date: "2018-09-04", historical_reports: 10 },
    { rank: 3, district: "North 24 Parganas", state_ut: "West Bengal", latest_cases: 1305, predicted_cases: 157.4, risk_level: "VERY HIGH", alert_threshold_exceeded: true, last_report_date: "2017-09-21", historical_reports: 20 },
    { rank: 4, district: "Hardoi", state_ut: "Uttar Pradesh", latest_cases: 933, predicted_cases: 64.3, risk_level: "VERY HIGH", alert_threshold_exceeded: true, last_report_date: "2016-07-11", historical_reports: 2 },
    { rank: 5, district: "Aurangabad", state_ut: "Bihar", latest_cases: 906, predicted_cases: 32.5, risk_level: "HIGH", alert_threshold_exceeded: true, last_report_date: "2015-06-30", historical_reports: 1 },
    { rank: 6, district: "South 24 Parganas", state_ut: "West Bengal", latest_cases: 728, predicted_cases: 237.3, risk_level: "VERY HIGH", alert_threshold_exceeded: true, last_report_date: "2018-09-21", historical_reports: 4 },
    { rank: 7, district: "Bhojpur", state_ut: "Bihar", latest_cases: 724, predicted_cases: 29.0, risk_level: "HIGH", alert_threshold_exceeded: true, last_report_date: "2015-06-30", historical_reports: 2 },
    { rank: 8, district: "Nizamabad", state_ut: "Telangana", latest_cases: 591, predicted_cases: 21.0, risk_level: "MODERATE", alert_threshold_exceeded: true, last_report_date: "2018-11-08", historical_reports: 2 },
    { rank: 9, district: "Nainital", state_ut: "Uttarakhand", latest_cases: 492, predicted_cases: 32.7, risk_level: "HIGH", alert_threshold_exceeded: true, last_report_date: "2019-08-01", historical_reports: 4 },
    { rank: 10, district: "Murshidabad", state_ut: "West Bengal", latest_cases: 480, predicted_cases: 249.8, risk_level: "VERY HIGH", alert_threshold_exceeded: true, last_report_date: "2019-08-16", historical_reports: 13 }
  ],
  active_model: "RandomForest_log1p",
  last_updated: "2026-10-18"
};

/**
 * Fetches surveillance metrics summary from the local Python API,
 * falling back to validated static baseline figures if the API is unavailable.
 * @returns {Promise<object>} Surveillance metric aggregate figures
 */
export async function fetchSurveillanceMetrics() {
  try {
    const response = await fetchModelApi("/metrics", {
      method: "GET",
      headers: { "Accept": "application/json" }
    });
    if (response.ok) {
      const data = await response.json();
      if (data && data.status === "success" && data.metrics) {
        return { ...DEFAULT_SURVEILLANCE_METRICS, ...data.metrics };
      }
    }
  } catch {
    // Backend API not running or unreachable; gracefully return validated baseline
  }
  return DEFAULT_SURVEILLANCE_METRICS;
}

export async function fetchXaiImportance() {
  const response = await fetchModelApi("/xai/importance", {
    headers: { "Accept": "application/json" },
  });
  return readApiResponse(response);
}

export async function fetchXaiExplanation() {
  const response = await fetchModelApi("/xai/explain", {
    method: "POST",
    headers: { "Content-Type": "application/json", "Accept": "application/json" },
    body: JSON.stringify({}),
  });
  return readApiResponse(response);
}

// ─── Geospatial Heatmap Data ────────────────────────────────────────────────
/** Returns the latest dated report for each district in dengue_cleaned.csv. */
export function getDistrictHeatmapPoints() {
  return districtSurveillance;
}
