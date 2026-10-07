// api.js - handles communication with the n8n webhook
import { N8N_WEBHOOK_URL } from "./config.js";

/**
 * Sends a prediction request.
 * @param {string} location - location string (e.g., "Pune")
 * @param {string} date - ISO date string (YYYY-MM-DD)
 * @returns {Promise<object>} API response JSON
 */
export async function fetchPrediction(location, date) {
  if (!N8N_WEBHOOK_URL) {
    throw new Error("Backend connection not configured. Please set N8N_WEBHOOK_URL in config.js.");
  }
  const payload = { location, date };
  const response = await fetch(N8N_WEBHOOK_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Network error: ${response.status} ${response.statusText}\n${text}`);
  }
  return response.json();
}
