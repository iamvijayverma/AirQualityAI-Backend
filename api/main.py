import os
import json
import logging
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI App
app = FastAPI(
    title="Urban Air Quality Intelligence API",
    description="Backend API for AQI Forecasting, Source Attribution, and Advisory Agents.",
    version="1.0.0"
)

# Global variables for models
forecast_model = None
source_model = None

# Constants
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
# Fallback if running directly in the root directory
if not os.path.exists(MODEL_DIR):
    MODEL_DIR = 'models'

FORECAST_MODEL_PATH = os.path.join(MODEL_DIR, 'aqi_model.pkl')
SOURCE_MODEL_PATH = os.path.join(MODEL_DIR, 'source_model.pkl')

# ==============================================================================
# PYDANTIC SCHEMAS
# ==============================================================================
class ForecastRequest(BaseModel):
    aqi_1: float = Field(..., description="AQI value 1 hour ago")
    aqi_6: float = Field(..., description="AQI value 6 hours ago")
    aqi_24: float = Field(..., description="AQI value 24 hours ago")

class SourceAttributionRequest(BaseModel):
    pm25: float = Field(..., description="Ambient PM2.5 Concentration")
    pm10: float = Field(..., description="Ambient PM10 Concentration")
    no2: float = Field(..., description="Gas Phase NO2 Tracer")
    traffic_density: float = Field(..., description="Traffic Density Index")
    construction_index: float = Field(..., description="Construction Velocity Index")
    industrial_index: float = Field(..., description="Industrial Operation Metric")

class CitizenAdvisoryRequest(BaseModel):
    aqi: int = Field(..., description="Current Air Quality Index")
    age: int = Field(..., description="Citizen's Age")
    condition: str = Field(..., description="Pre-existing Medical Conditions")
    language: str = Field(default="English", description="Target Language (English/Hindi)")

class EnforcementRequest(BaseModel):
    aqi: int = Field(..., description="Current Air Quality Index")
    traffic_pct: float = Field(..., description="Traffic Sector Contribution Percentage")
    construction_pct: float = Field(..., description="Construction Sector Contribution Percentage")
    industry_pct: float = Field(..., description="Industrial Sector Contribution Percentage")

# ==============================================================================
# LIFESPAN / STARTUP
# ==============================================================================
@app.on_event("startup")
async def load_models():
    global forecast_model, source_model
    try:
        if os.path.exists(FORECAST_MODEL_PATH):
            forecast_model = joblib.load(FORECAST_MODEL_PATH)
            logger.info(f"Loaded Forecast Model from {FORECAST_MODEL_PATH}")
        else:
            logger.warning("Forecast Model not found. Endpoints will use rule-based fallbacks.")

        if os.path.exists(SOURCE_MODEL_PATH):
            source_model = joblib.load(SOURCE_MODEL_PATH)
            logger.info(f"Loaded Source Attribution Model from {SOURCE_MODEL_PATH}")
        else:
            logger.warning("Source Attribution Model not found. Endpoints will use rule-based fallbacks.")
    except Exception as e:
        logger.error(f"Error loading machine learning models: {e}")

# ==============================================================================
# ENDPOINTS
# ==============================================================================
@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "Welcome to the AI-Powered Urban Air Quality Intelligence Platform API"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "forecast_model_loaded": forecast_model is not None,
        "source_model_loaded": source_model is not None
    }

@app.post("/forecast")
async def predict_aqi(request: ForecastRequest):
    features = np.array([[request.aqi_1, request.aqi_6, request.aqi_24]])
    
    if forecast_model:
        try:
            prediction = forecast_model.predict(features)[0]
            return {"target_hour": "+1H", "predicted_aqi": round(float(prediction), 2), "method": "RandomForest"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model prediction error: {str(e)}")
    else:
        # Fallback heuristic
        pred = (request.aqi_1 * 0.6) + (request.aqi_6 * 0.3) + (request.aqi_24 * 0.1)
        return {"target_hour": "+1H", "predicted_aqi": round(pred, 2), "method": "Heuristic_Fallback"}

@app.post("/source-attribution")
async def attribute_source(request: SourceAttributionRequest):
    features = np.array([[
        request.pm25, request.pm10, request.no2, 
        request.traffic_density, request.construction_index, request.industrial_index
    ]])
    
    if source_model:
        try:
            predicted_class = source_model.predict(features)[0]
            probabilities = source_model.predict_proba(features)[0]
            classes = source_model.classes_
            
            prob_dict = {str(cls): round(float(prob), 4) for cls, prob in zip(classes, probabilities)}
            
            return {
                "primary_source": predicted_class,
                "probabilities": prob_dict,
                "method": "RandomForestClassifier"
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model prediction error: {str(e)}")
    else:
        # Fallback deterministic rules
        scores = {
            'Traffic': request.traffic_density * 1.2 + request.no2 * 0.8,
            'Construction': request.construction_index * 2.0 + (request.pm10 - request.pm25) * 0.5,
            'Industry': request.industrial_index * 1.5 + request.pm25 * 0.5,
            'Mixed': (request.traffic_density + request.construction_index + request.industrial_index) * 0.6
        }
        total_score = sum(scores.values())
        prob_dict = {k: round(v / total_score, 4) for k, v in scores.items()}
        primary = max(scores, key=scores.get)
        
        return {
            "primary_source": primary,
            "probabilities": prob_dict,
            "method": "Heuristic_Fallback"
        }

@app.post("/citizen-advisory")
async def get_advisory(request: CitizenAdvisoryRequest):
    is_hindi = request.language.lower() == "hindi"
    is_vulnerable = "asthma" in request.condition.lower() or "heart" in request.condition.lower()

    if request.aqi > 300 or (request.aqi > 200 and is_vulnerable):
        risk = "Severe"
        primary = "हवा की गुणवत्ता बहुत खराब है। कृपया घर के अंदर रहें।" if is_hindi else "Air quality is severely degraded. Please remain indoors."
        precautions = [
            "बाहर जाते समय N95 मास्क पहनें।" if is_hindi else "Wear an N95 mask if you must go outside.",
            "सभी बाहरी व्यायाम बंद करें।" if is_hindi else "Stop all outdoor physical exertion."
        ]
        mask = True
    elif request.aqi <= 100:
        risk = "Low"
        primary = "हवा की गुणवत्ता अच्छी है।" if is_hindi else "Air quality is good. Safe for normal activities."
        precautions = [
            "आप सुरक्षित रूप से बाहर व्यायाम कर सकते हैं।" if is_hindi else "You may safely exercise outdoors."
        ]
        mask = False
    else:
        risk = "Moderate"
        primary = "हवा की गुणवत्ता मध्यम है।" if is_hindi else "Air quality is moderate. Exercise caution."
        precautions = [
            "भारी बाहरी काम कम करें।" if is_hindi else "Reduce heavy outdoor exertion."
        ]
        mask = is_vulnerable
        
    return {
        "risk_level": risk,
        "primary_advisory": primary,
        "precautions": precautions,
        "mask_recommended": mask
    }

@app.post("/enforcement")
async def generate_enforcement(request: EnforcementRequest):
    risk = "Severe" if request.aqi > 300 else ("High" if request.aqi > 200 else "Moderate")
    actions = []
    reduction_estimate = 0
    
    if request.traffic_pct > 50:
        actions.extend(["Implement odd-even vehicle restrictions.", "Restrict heavy diesel trucks."])
        reduction_estimate += 35
    if request.construction_pct > 30:
        actions.extend(["Mandate continuous dust suppression.", "Deploy site inspections."])
        reduction_estimate += 20
    if request.industry_pct > 40:
        actions.extend(["Initiate emission audits.", "Enforce temporary operational rollbacks."])
        reduction_estimate += 45

    if not actions:
        actions.append("Maintain routine patrol sweeps.")
        reduction_estimate = 5

    base_aqi_factor = min((request.aqi / 500) * 50, 50)
    max_sector_factor = min(max(request.traffic_pct, request.construction_pct, request.industry_pct) * 0.5, 50)
    score = int(base_aqi_factor + max_sector_factor)

    return {
        "risk_level": risk,
        "priority_actions": actions,
        "expected_aqi_reduction": f"Estimated reduction of {reduction_estimate - 5}-{reduction_estimate + 10} AQI points.",
        "enforcement_score": min(score, 100)
    }

if __name__ == "__main__":
    import uvicorn
    # Allows the script to be run directly via `python api/main.py`
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)