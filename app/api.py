"""
FastAPI application for DeepMatch AI prediction service.
Loads trained model and provides REST API for match predictions.
"""

import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List

import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI(
    title="DeepMatch AI",
    description="Deep learning model for Premier League match outcome prediction",
    version="1.0.0"
)

# Configuration
MODEL_PATH = Path("models/deepmatch_model.keras")
SCALER_PATH = Path("models/scaler.pkl")
TEAM_ENCODER_PATH = Path("models/team_encoder.pkl")
NUM_COLS_PATH = Path("models/num_cols.pkl")

# Global variables for loaded model and preprocessors
model = None
scaler = None
team_encoder = None
num_cols = None


def load_artifacts():
    """Load trained model and preprocessing artifacts."""
    global model, scaler, team_encoder, num_cols
    
    try:
        # Load model
        model = tf.keras.models.load_model(str(MODEL_PATH))
        print(f"✓ Model loaded from {MODEL_PATH}")
        
        # Load preprocessors
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        print(f"✓ Scaler loaded from {SCALER_PATH}")
        
        with open(TEAM_ENCODER_PATH, "rb") as f:
            team_encoder = pickle.load(f)
        print(f"✓ Team encoder loaded from {TEAM_ENCODER_PATH}")
        
        with open(NUM_COLS_PATH, "rb") as f:
            num_cols = pickle.load(f)
        print(f"✓ Numerical columns loaded from {NUM_COLS_PATH}")
        
    except FileNotFoundError as e:
        raise RuntimeError(f"Missing required artifact: {e}")
    except Exception as e:
        raise RuntimeError(f"Error loading artifacts: {e}")


# Load artifacts at startup
load_artifacts()


# ============================================================================
# Request/Response Models
# ============================================================================

class PredictionRequest(BaseModel):
    """Input schema for prediction endpoint."""
    home_team: str
    away_team: str
    B365H: float
    B365D: float
    B365A: float
    home_avg_goals_for: float
    home_avg_goals_against: float
    home_avg_points: float
    away_avg_goals_for: float
    away_avg_goals_against: float
    away_avg_points: float
    form_points_diff: float
    form_goals_for_diff: float
    form_goals_against_diff: float


class PredictionResponse(BaseModel):
    """Output schema for prediction endpoint."""
    predicted_class: str
    predicted_class_code: int
    home_win_probability: float
    draw_probability: float
    away_win_probability: float
    confidence: float
    home_team: str
    away_team: str


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/", tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        dict: Service status information
    """
    return {
        "status": "healthy",
        "service": "DeepMatch AI Prediction API",
        "version": "1.0.0",
        "model": "Neural Network with Team Embeddings"
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Predict match outcome for a given match scenario.
    
    Args:
        request: PredictionRequest containing team names, odds, and form metrics
        
    Returns:
        PredictionResponse: Predicted class and probabilities
        
    Raises:
        HTTPException: If prediction fails or teams are unknown
    """
    try:
        # ====================================================================
        # Encode teams
        # ====================================================================
        known_teams = team_encoder.classes_.tolist()
        
        # Check if teams are known
        if request.home_team not in known_teams:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown home team: {request.home_team}. Known teams: {sorted(known_teams)}"
            )
        
        if request.away_team not in known_teams:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown away team: {request.away_team}. Known teams: {sorted(known_teams)}"
            )
        
        # Encode team names to integers
        home_team_encoded = team_encoder.transform([request.home_team])[0]
        away_team_encoded = team_encoder.transform([request.away_team])[0]
        
        # ====================================================================
        # Prepare numerical features
        # ====================================================================
        numerical_features = np.array([
            [
                request.B365H,
                request.B365D,
                request.B365A,
                request.home_avg_goals_for,
                request.home_avg_goals_against,
                request.home_avg_points,
                request.away_avg_goals_for,
                request.away_avg_goals_against,
                request.away_avg_points,
                request.form_points_diff,
                request.form_goals_for_diff,
                request.form_goals_against_diff,
            ]
        ])
        
        # Scale numerical features using training scaler
        numerical_features_scaled = scaler.transform(numerical_features)
        
        # ====================================================================
        # Make prediction
        # ====================================================================
        # Model expects: [numerical_input, home_team_input, away_team_input]
        predictions = model.predict(
            [
                numerical_features_scaled,
                np.array([home_team_encoded]),
                np.array([away_team_encoded])
            ],
            verbose=0
        )
        
        # predictions shape: (1, 3) with probabilities for [A, D, H]
        probabilities = predictions[0]
        
        # Class mapping: A=0, D=1, H=2
        away_prob = float(probabilities[0])
        draw_prob = float(probabilities[1])
        home_prob = float(probabilities[2])
        
        # Get predicted class (argmax)
        predicted_class_code = int(np.argmax(probabilities))
        class_names = {0: "A", 1: "D", 2: "H"}
        predicted_class = class_names[predicted_class_code]
        
        # Confidence is the max probability
        confidence = float(np.max(probabilities))
        
        # ====================================================================
        # Return response
        # ====================================================================
        return PredictionResponse(
            predicted_class=predicted_class,
            predicted_class_code=predicted_class_code,
            home_win_probability=home_prob,
            draw_probability=draw_prob,
            away_win_probability=away_prob,
            confidence=confidence,
            home_team=request.home_team,
            away_team=request.away_team
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@app.get("/teams", tags=["Metadata"])
async def get_known_teams():
    """
    Get list of all known teams in the encoder.
    
    Returns:
        dict: List of known teams
    """
    known_teams = sorted(team_encoder.classes_.tolist())
    return {
        "teams": known_teams,
        "count": len(known_teams)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
