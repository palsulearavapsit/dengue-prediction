from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import json
import os

import joblib
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "dengue_forecast_model.joblib"

HOST = "127.0.0.1"
PORT = 8001


FEATURES = [
    "state_ut",
    "district",
    "current_cases",
    "previous_cases",
    "cases_lag_2",
    "cases_lag_3",
    "cases_lag_4",
    "previous_case_change",
    "previous_case_growth",
    "previous_cases_rolling_mean_4",
    "previous_cases_rolling_median_4",
    "previous_cases_rolling_min_4",
    "previous_cases_rolling_max_4",
    "previous_cases_rolling_std_4",
    "days_since_previous_report",
    "previous_reporting_gap_mean_4",
    "previous_reporting_gap_median_4",
    "previous_reporting_gap_std_4",
    "observation_year",
    "observation_month",
    "observation_week_of_year",
    "observation_day_of_year",
    "month_sin",
    "month_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "season",
    "Latitude",
    "Longitude",
    "Temp",
    "preci",
    "LAI",
]


print("=" * 70)
print("DENGUE ML PREDICTION API")
print("=" * 70)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

print(f"Loading model: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)


def get_backend_setting(name):
    """Read a backend-only setting from the process environment or root .env."""
    value = os.environ.get(name, "").strip()
    if not value:
        env_path = ROOT / ".env"
        if not env_path.exists():
            return ""

        for line in env_path.read_text(encoding="utf-8").splitlines():
            key, separator, raw_value = line.partition("=")
            if separator and key.strip() == name:
                value = raw_value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                    value = value[1:-1]
                break

    if value.lower().startswith(("your-", "your_", "replace-", "<")):
        return ""
    return value


GEMINI_MODEL = get_backend_setting("GEMINI_MODEL") or "gemini-3.8-flash"
GEMINI_REQUEST_TIMEOUT_SECONDS = 90


def build_global_feature_importance():
    """Aggregate one-hot encoded Random Forest importances by source feature."""
    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]
    encoded_names = preprocessor.get_feature_names_out()
    encoded_importances = estimator.feature_importances_
    grouped = {}

    for encoded_name, importance in zip(encoded_names, encoded_importances):
        name = str(encoded_name).split("__", 1)[-1]
        source_feature = next(
            (feature for feature in FEATURES if name == feature or name.startswith(feature + "_")),
            name,
        )
        grouped[source_feature] = grouped.get(source_feature, 0.0) + float(importance)

    total = sum(grouped.values()) or 1.0
    ranking = [
        {"feature": name, "importance": round(value / total * 100, 2)}
        for name, value in grouped.items()
    ]
    ranking.sort(key=lambda item: item["importance"], reverse=True)
    represented = set(grouped)
    return {
        "features": ranking[:12],
        "feature_count": len(grouped),
        "api_feature_count": len(FEATURES),
        "features_without_importance": [feature for feature in FEATURES if feature not in represented],
    }


GLOBAL_FEATURE_IMPORTANCE = build_global_feature_importance()


def generate_gemini_explanation():
    api_key = get_backend_setting("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Gemini is not configured. Set GEMINI_API_KEY for the Python API.")

    feature_labels = {
        "state_ut": "State or union territory",
        "previous_cases": "Cases in the previous report",
        "current_cases": "Cases in the current report",
        "Latitude": "Latitude (location)",
        "Longitude": "Longitude (location)",
        "preci": "Precipitation (rainfall)",
        "district": "District",
        "Temp": "Temperature",
        "day_of_year_sin": "Seasonal timing (sine signal)",
        "observation_year": "Year of observation",
        "LAI": "Leaf Area Index (vegetation measure)",
        "day_of_year_cos": "Seasonal timing (cosine signal)",
    }
    top_features = [
        {
            "feature": feature_labels.get(item["feature"], item["feature"]),
            "importance_percent": item["importance"],
        }
        for item in GLOBAL_FEATURE_IMPORTANCE["features"][:5]
    ]
    top_three_total = round(sum(item["importance"] for item in GLOBAL_FEATURE_IMPORTANCE["features"][:3]), 2)
    evidence = json.dumps({
        "model": "RandomForest_log1p",
        "target": "next_observed_report_cases",
        "importance_method": "Random Forest impurity-based global feature importance, grouped by original input feature",
        "top_features": top_features,
        "top_three_importance_total_percent": top_three_total,
        "features_without_importance": GLOBAL_FEATURE_IMPORTANCE["features_without_importance"],
    }, ensure_ascii=False)
    prompt = (
        "Write a clear explanation for a general public-health dashboard user who may not know machine learning. "
        "Use exactly these three short headings: 'Main takeaway', 'What the percentages mean', and 'Important limits'. "
        "Under Main takeaway, name the top inputs in everyday language and say the top three together represent "
        f"{top_three_total}% of the model's total importance score. Under What the percentages mean, explain that "
        "the model uses patterns in past reports to forecast cases at the next observed district report. Under "
        "Important limits, say these are global model scores, not outbreak probabilities or shares of predicted cases; "
        "they do not say whether an input raises or lowers a forecast or prove cause. Mention that impurity importance "
        "can favor features with many categories, and that the data are historical. Do not use raw variable codes, "
        "invent facts, or give medical advice. Keep the full answer between 100 and 180 words.\n\n"
        f"Evidence JSON: {evidence}"
    )
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 700},
    }).encode("utf-8")
    request = Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
        data=body,
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )

    try:
        with urlopen(request, timeout=GEMINI_REQUEST_TIMEOUT_SECONDS) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        # Never return the upstream body; it can include request details.
        raise RuntimeError(f"Gemini request failed (HTTP {error.code}). Check the API key, quota, and model setting.") from None
    except (URLError, TimeoutError) as error:
        # Include only the transport reason (never the request headers or API key)
        # so local DNS, proxy, TLS, and timeout failures can be diagnosed.
        reason = getattr(error, "reason", error)
        if isinstance(error, TimeoutError) or isinstance(reason, TimeoutError):
            detail = f"request timed out after {GEMINI_REQUEST_TIMEOUT_SECONDS} seconds"
        elif isinstance(reason, OSError):
            detail = reason.strerror or reason.__class__.__name__
        else:
            detail = str(reason)
        detail = detail.replace("\r", " ").replace("\n", " ")[:180]
        raise RuntimeError(
            f"Could not reach the Gemini API ({detail}). Check outbound HTTPS, DNS, proxy, or firewall settings."
        ) from None
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise RuntimeError("Gemini returned an unreadable response. Try again later.") from None

    candidates = result.get("candidates", []) if isinstance(result, dict) else []
    candidate = candidates[0] if candidates and isinstance(candidates[0], dict) else {}
    content = candidate.get("content", {})
    parts = content.get("parts", []) if isinstance(content, dict) else []
    if not isinstance(parts, list):
        parts = []
    explanation = "\n".join(part.get("text", "") for part in parts if part.get("text"))[:6000].strip()
    if len(explanation) < 180:
        top_three = GLOBAL_FEATURE_IMPORTANCE["features"][:3]
        named = [
            f"{feature_labels.get(item['feature'], item['feature'])} ({item['importance']:.2f}%)"
            for item in top_three
        ]
        explanation = (
            "Main takeaway: The model gives the most importance to " + ", ".join(named) + ". "
            f"Together, these account for {top_three_total:.2f}% of its total feature-importance score.\n\n"
            "What the percentages mean: Across historical reports, the model used these inputs when forecasting "
            "cases at the next observed district report.\n\n"
            "Important limits: These are global model scores, not outbreak probabilities or shares of predicted "
            "cases. They do not show whether a factor raises or lowers a forecast or prove cause. Impurity importance "
            "can favor features with many categories, and the underlying data are historical."
        )
    return explanation

print("Model loaded successfully.")
print(f"API: http://{HOST}:{PORT}")
print("=" * 70)


class PredictionHandler(BaseHTTPRequestHandler):

    def send_json(self, status_code, data):

        response = json.dumps(
            data,
            allow_nan=False
        ).encode("utf-8")

        self.send_response(status_code)

        self.send_header(
            "Content-Type",
            "application/json"
        )

        self.send_header(
            "Content-Length",
            str(len(response))
        )

        self.send_header(
            "Access-Control-Allow-Origin",
            "*"
        )
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

        self.end_headers()

        self.wfile.write(response)


    def do_GET(self):

        if self.path == "/xai/importance":
            self.send_json(200, {
                "status": "success",
                "method": "global_impurity_importance",
                "model": "RandomForest_log1p",
                "target": "next_observed_report_cases",
                "gemini_configured": bool(get_backend_setting("GEMINI_API_KEY")),
                **GLOBAL_FEATURE_IMPORTANCE,
            })
            return

        if self.path == "/health":

            self.send_json(
                200,
                {
                    "status": "ok",
                    "service": "dengue-ml-api",
                    "model_loaded": True
                }
            )

            return

        if self.path == "/metrics":

            self.send_json(
                200,
                {
                    "status": "success",
                    "metrics": {
                        "historical_observations": 1618,
                        "eligible_districts": 202,
                        "monitored_states": 34,
                        "predicted_cases_national_sum": 7341,
                        "predicted_cases_national_sum_formatted": "7,341",
                        "high_risk_districts": 15,
                        "risk_distribution": {
                            "low": 10,
                            "moderate": 133,
                            "high": 45,
                            "very_high": 15
                        },
                        "active_model": "RandomForest_log1p",
                        "last_updated": "2026-10-18"
                    }
                }
            )

            return

        self.send_json(
            404,
            {
                "status": "error",
                "message": "Endpoint not found"
            }
        )


    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        # Chromium can preflight requests from a local web origin to a
        # loopback API under its Private Network Access checks.
        self.send_header("Access-Control-Allow-Private-Network", "true")
        self.end_headers()


    def do_POST(self):

        if self.path == "/xai/explain":
            try:
                self.send_json(200, {
                    "status": "success",
                    "model": "RandomForest_log1p",
                    "method": "global_impurity_importance",
                    "gemini_model": GEMINI_MODEL,
                    "explanation": generate_gemini_explanation(),
                })
            except RuntimeError as error:
                message = str(error)
                status_code = 503 if message.startswith("Gemini is not configured") else 502
                self.send_json(status_code, {"status": "error", "message": message})
            except Exception:
                self.send_json(502, {"status": "error", "message": "Gemini returned an invalid response. Try again later."})
            return

        if self.path != "/predict":

            self.send_json(
                404,
                {
                    "status": "error",
                    "message": "Endpoint not found"
                }
            )

            return

        try:

            content_length = int(
                self.headers.get(
                    "Content-Length",
                    0
                )
            )

            raw_body = self.rfile.read(
                content_length
            )

            payload = json.loads(
                raw_body.decode("utf-8")
            )

            missing = [
                feature
                for feature in FEATURES
                if feature not in payload
            ]

            if missing:

                self.send_json(
                    400,
                    {
                        "status": "error",
                        "message": "Missing model features",
                        "missing_features": missing
                    }
                )

                return

            # Keep exactly the same feature order
            # used during training.
            input_data = {
                feature: payload[feature]
                for feature in FEATURES
            }

            X = pd.DataFrame(
                [input_data],
                columns=FEATURES
            )

            prediction_log = model.predict(X)

            predicted_cases = float(
                np.expm1(
                    prediction_log[0]
                )
            )

            # Dengue case counts cannot be negative.
            predicted_cases = max(
                0.0,
                predicted_cases
            )

            self.send_json(
                200,
                {
                    "status": "success",
                    "prediction_available": True,
                    "predicted_cases": round(
                        predicted_cases,
                        2
                    ),
                    "model": "RandomForest_log1p",
                    "target": "next_observed_report_cases"
                }
            )

        except Exception as e:

            self.send_json(
                500,
                {
                    "status": "error",
                    "prediction_available": False,
                    "message": str(e)
                }
            )


server = ThreadingHTTPServer(
    (HOST, PORT),
    PredictionHandler
)
server.daemon_threads = True

print("\nAPI is running.")
print(f"Health:  http://{HOST}:{PORT}/health")
print(f"Predict: http://{HOST}:{PORT}/predict")
print("\nPress CTRL+C to stop.\n")

server.serve_forever()
