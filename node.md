# Dengue Prediction Workflow — Node-by-Node Technical Explanation

## 1. Webhook
**Purpose:** Receives the user's prediction request.
**Input example:**
```json
{
  "location": "Pune",
  "date": "2022-04-23"
}
```
**Output:** Passes the request into the workflow.
**URL:** `http://localhost:5678/webhook-test/dengue-prediction`

## 2. Location Resolver
**Purpose:** Converts the entered city or district name into geographic coordinates.
**Input:** `Pune`
**Output example:**
```json
{
  "latitude": 18.521428,
  "longitude": 73.8544541
}
```
**API:** OpenStreetMap Nominatim
**Website:** https://nominatim.openstreetmap.org/

## 3. Location Normalizer
**Purpose:** Standardizes the resolved location into consistent fields.
**Input:** Location lookup response.
**Output example:**
```json
{
  "city": "Pune",
  "district": "Pune",
  "state": "Maharashtra",
  "latitude": 18.521428,
  "longitude": 73.8544541
}
```
**Source:** Previous node; no separate external API.

## 4. Historical Weather Data
**Purpose:** Retrieves historical weather information for the selected location and date.
**Input:** Latitude, longitude, and date range.
**Output:** Historical temperature, humidity, and precipitation information.
**API:** Open-Meteo Historical Weather API
**Link:** https://archive-api.open-meteo.com/v1/archive
**Documentation:** https://open-meteo.com/en/docs/historical-weather-api

## 5. Dengue Surveillance Data
**Purpose:** Attempts to retrieve dengue surveillance information, such as reported cases and deaths.
**Input:** Configured surveillance-data request.
**Output:** Surveillance response, if available.
**Source:** National Centre for Vector Borne Diseases Control (NCVBDC)
**Website:** https://ncvbdc.mohfw.gov.in/
**Note:** The configured endpoint timed out during earlier tests, so the fallback branch may be used.

## 6. Code (Fallback)
**Purpose:** Handles an unsuccessful surveillance request and prepares fallback data.
**Input:** Fallback-path data.
**Output:** Structured data for the following HTTP Request node.
**Source:** Internal JavaScript; no separate API.

## 7. HTTP Request
**Purpose:** Fetches data from the external endpoint configured in the node.
**Input:** Prepared request parameters.
**Output:** HTTP response data.
**URL:** Uses the endpoint configured in the node. Verify the node's URL field to identify the exact service.

## 8. Code in JavaScript
**Purpose:** Processes the retrieved response and handles missing surveillance data.
**Input:** HTTP response.
**Output:** Cleaned or normalized JSON fields for downstream processing.
**Source:** Internal JavaScript; no separate API.

## 9. Feature Builder
**Purpose:** Combines available weather, location, and dengue information into structured features.
**Input:** Processed environmental and location data.
**Output:** A consistent JSON object containing available feature values.
**Source:** Previous workflow nodes.

## 10. Temporal Feature Builder
**Purpose:** Generates calendar-related features from the selected observation date.
**Input:** Date and existing feature data.
**Output example:**
```json
{
  "year": 2022,
  "month": 4,
  "week_of_year": 17,
  "season": 2
}
```
**Source:** Internal JavaScript; no separate API.

## 11. Historical Dengue Data
**Purpose:** Reads historical dengue records from a local CSV file.
**Input:** File path.
**Output:** CSV file contents for the extraction node.
**File:** `Final_data(3).csv`
**Location:** Project folder.

## 12. Extract from File
**Purpose:** Parses the CSV into individual records that can be filtered and processed.
**Input:** CSV file data.
**Output:** Structured records, typically one item per row.
**Source:** Local file; no external API.

## 13. Filter Dengue Data
**Purpose:** Keeps relevant dengue observations for the selected district and state.
**Input:** Historical dataset records.
**Output example:**
```json
{
  "district": "Pune",
  "state_ut": "Maharashtra",
  "Disease": "Dengue",
  "Cases": "21",
  "day": "16",
  "mon": "5",
  "year": "2022"
}
```
**Source:** Local CSV and n8n filtering logic.

## 14. Dengue Lag Feature Builder
**Purpose:** Selects observations on or before the requested date and calculates historical case lags, rolling statistics, reporting gaps, and temporal features.
**Input:** Filtered dengue records and requested date.
**Output:** Model features such as:
- `current_cases`
- `previous_cases`
- `cases_lag_2`
- `previous_cases_rolling_mean_4`
- `days_since_previous_report`
- `observation_month`
- `Temp`, `preci`, and `LAI`
**Source:** Internal JavaScript and historical CSV data.

## 15. Prediction Feature Builder
**Purpose:** Organizes the generated feature values into the format expected by the prediction pipeline.
**Input:** Historical and temporal features.
**Output:** Structured machine-learning feature object.
**Source:** Internal JavaScript.

## 16. Feature Validation
**Purpose:** Checks whether the required model features are present and usable.
**Input:** Prepared feature object.
**Output example:**
```json
{
  "prediction_ready": true
}
```
**Source:** Internal validation logic.

## 17. Prediction Readiness Router
**Purpose:** Chooses the success or failure branch based on feature availability.
**Input:** Feature-validation result.
**Output:** Routes to prediction preparation or the data-unavailable branch.
**Source:** n8n conditional routing.

## 18. Prepare ML Input
**Purpose:** Builds the JSON request expected by the Python machine-learning API.
**Input:** Validated model features.
**Output:** An `ml_input` object containing the model's required features.
**Source:** Internal JavaScript.

## 19. HTTP Request1
**Purpose:** Sends the prepared feature values to the trained Random Forest model API.
**Input:** `ml_input` JSON.
**Output:** Predicted case count and model metadata.
**API:** Local Python prediction service.
**Endpoint:** `http://127.0.0.1:8000/predict`

## 20. Code in JavaScript1
**Purpose:** Formats the model prediction into a final success response.
**Input:** Python API response.
**Output example:**
```json
{
  "status": "success",
  "message": "The model predicts approximately 17.58 dengue cases for the next observed district report."
}
```
**Source:** Internal JavaScript.

## 21. Respond to Webhook
**Purpose:** Sends the final result back to the user who submitted the request.
**Input:** Final response object.
**Output:** JSON containing only `status` and `message`.
**Endpoint:** `http://localhost:5678/webhook-test/dengue-prediction`

## 22. Data Unavailable Response
**Purpose:** Prepares a response when insufficient historical data or valid features prevent a prediction.
**Input:** Data-availability or feature-validation failure.
**Output:** Data-unavailable response object.
**Source:** Internal n8n configuration.

## 23. Code in JavaScript2
**Purpose:** Formats the data-unavailable response before it is returned to the requester.
**Input:** Failure-branch data.
**Output:** Structured failure response.
**Source:** Internal JavaScript.

---

# Overall Workflow Summary

1. Receive a location and optional date.
2. Resolve the location into geographic coordinates.
3. Retrieve available weather and surveillance information.
4. Load and filter historical dengue records.
5. Generate historical, lag, rolling, temporal, and environmental features.
6. Validate the required features.
7. Send the features to the Random Forest prediction API.
8. Return a concise success or data-unavailable message.

**Machine-learning algorithm:** Random Forest (`RandomForest_log1p`)

**Prediction target:** `next_observed_report_cases`

**Important limitation:** The model predicts cases at the next observed district report, not necessarily on the exact requested date. The current Pune dataset's latest observation is 16 May 2022, so requesting a later date does not automatically provide updated dengue surveillance data.

**API-link note:** The links above identify the documented Open-Meteo and Nominatim services. For the generic HTTP Request node and NCVBDC request, inspect the actual URL field in n8n before documenting an exact endpoint.
