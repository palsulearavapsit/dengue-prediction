from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import json

import joblib
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "dengue_forecast_model.joblib"

HOST = "127.0.0.1"
PORT = 8000


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

        self.end_headers()

        self.wfile.write(response)


    def do_GET(self):

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

        self.send_json(
            404,
            {
                "status": "error",
                "message": "Endpoint not found"
            }
        )


    def do_POST(self):

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


server = HTTPServer(
    (HOST, PORT),
    PredictionHandler
)

print("\nAPI is running.")
print("Health:  http://127.0.0.1:8000/health")
print("Predict: http://127.0.0.1:8000/predict")
print("\nPress CTRL+C to stop.\n")

server.serve_forever()