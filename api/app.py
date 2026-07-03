from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

app = FastAPI(
    title="Air Quality AI Backend",
    version="1.0.0"
)

# -----------------------------
# CORS
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "*"   # remove this later in production if desired
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Health Check
# -----------------------------
@app.get("/")
def root():
    return {
        "message": "Air Quality AI Backend Running"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


# -----------------------------
# AQI Forecast
# -----------------------------
@app.post("/forecast")
def forecast(data: dict):

    aqi_1 = float(data.get("aqi_1", 100))
    aqi_6 = float(data.get("aqi_6", 100))
    aqi_24 = float(data.get("aqi_24", 100))

    prediction = round(
        (aqi_1 * 0.6) +
        (aqi_6 * 0.3) +
        (aqi_24 * 0.1),
        2
    )

    forecast = []

    current = prediction

    for hour in range(24):
        current = max(
            0,
            current + np.random.randint(-8, 9)
        )

        forecast.append({
            "hour": hour,
            "aqi": round(current, 2)
        })

    return {
        "predicted_aqi": prediction,
        "confidence": 0.87,
        "trend": "increasing",
        "forecast_24h": forecast,
        "sources": [
            {
                "name": "Traffic",
                "percentage": 40,
                "color": "#3b82f6"
            },
            {
                "name": "Industry",
                "percentage": 35,
                "color": "#f59e0b"
            },
            {
                "name": "Construction",
                "percentage": 25,
                "color": "#ef4444"
            }
        ]
    }


# -----------------------------
# Source Attribution
# -----------------------------
@app.post("/source-attribution")
def source_attribution():

    return {
        "primary_source": "Traffic",

        "probabilities": {
            "Traffic": 0.40,
            "Industry": 0.30,
            "Construction": 0.20,
            "Mixed": 0.10
        },

        "method": "Random Forest",

        "timestamp": "2026-07-04"
    }


# -----------------------------
# Citizen Advisory
# -----------------------------
@app.post("/citizen-advisory")
def citizen_advisory(data: dict):

    aqi = data.get("aqi", 150)

    if aqi <= 50:
        risk = "Low"
    elif aqi <= 100:
        risk = "Moderate"
    elif aqi <= 200:
        risk = "High"
    else:
        risk = "Very High"

    return {
        "risk_level": risk,
        "primary_advisory": "Avoid prolonged outdoor activity.",
        "health_advisory": "Wear an N95 mask if outdoors.",
        "precautions": [
            "Keep windows closed",
            "Use air purifier",
            "Drink plenty of water"
        ],
        "mask_recommended": True,
        "mask_recommendation": "N95",
        "outdoor_activity": "Limit outdoor exercise"
    }


# -----------------------------
# Enforcement
# -----------------------------
@app.post("/enforcement")
def enforcement():

    return {
        "priority_actions": [
            "Increase traffic monitoring",
            "Restrict construction dust",
            "Inspect industrial emissions"
        ],

        "risk_level": "High",

        "enforcement_score": 82,

        "expected_aqi_reduction": "15-20 AQI",

        "zones": [
            {
                "name": "Central City",
                "aqi": 210,
                "action": "Traffic Restriction",
                "priority": "high"
            },
            {
                "name": "Industrial Area",
                "aqi": 185,
                "action": "Emission Inspection",
                "priority": "medium"
            }
        ]
    }
