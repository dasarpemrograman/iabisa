"""
FASKES PREDICTION API
=====================
FastAPI endpoint for healthcare facility predictions.
Downloads models from Supabase storage and returns predictions.

Usage:
    uvicorn api:app --reload
    
Endpoints:
    POST /predict - Generate predictions
    GET /health - Health check
    GET /models - List available models

Author: Data Science Team
Date: November 2025
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import tempfile
import shutil

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))
from supabase_storage import SupabaseModelStorage, download_faskes_model

# Load environment
load_dotenv()

app = FastAPI(
    title="Healthkathon Universal Prediction API",
    description="Unified healthcare prediction API supporting Faskes, Penyakit, and Peserta with Supabase integration",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global cache for models (to avoid re-downloading)
MODEL_CACHE = {}
TEMP_DIR = Path(tempfile.gettempdir()) / "faskes_models"
TEMP_DIR.mkdir(exist_ok=True)

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class PredictionRequest(BaseModel):
    """Request model for prediction endpoint."""
    
    dataset: str = Field(
        ..., 
        description="Dataset to predict: 'fkrtl', 'klinik_pratama', or 'praktek_dokter'",
        example="fkrtl"
    )
    start_year: int = Field(
        ..., 
        description="Starting year for predictions",
        ge=2019, 
        le=2030,
        example=2019
    )
    n_years: int = Field(
        default=1,
        description="Number of years to predict",
        ge=1,
        le=10,
        example=3
    )
    provinces: Optional[List[str]] = Field(
        default=None,
        description="Optional list of provinces to filter. If None, predicts all provinces.",
        example=["DKI Jakarta", "Jawa Barat"]
    )
    use_cache: bool = Field(
        default=True,
        description="Whether to use cached model (faster) or download fresh from Supabase",
        example=True
    )


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    
    success: bool
    dataset: str
    start_year: int
    end_year: int
    predictions: List[Dict[str, Any]]
    total_facilities: int
    model_info: Dict[str, Any]
    timestamp: str


class PenyakitRequest(BaseModel):
    """Request model for penyakit (disease) predictions."""
    
    service_type: Optional[str] = Field(
        None,
        description="Service type: 'RITL', 'RITP', 'RJTL', 'RJTP'. If None, predicts all services.",
        example="RITL"
    )
    top_n: int = Field(
        default=10,
        description="Number of top diseases to predict",
        ge=1,
        le=50,
        example=10
    )
    months: int = Field(
        default=12,
        description="Number of months to predict",
        ge=1,
        le=60,
        example=12
    )
    use_cache: bool = Field(
        default=True,
        description="Whether to use cached models",
        example=True
    )


class PesertaRequest(BaseModel):
    """Request model for peserta (participant) predictions."""
    
    segment_type: str = Field(
        ...,
        description="Segment type: 'geo', 'kelas', or 'segmen'",
        example="geo"
    )
    months: int = Field(
        default=12,
        description="Number of months to predict",
        ge=1,
        le=60,
        example=12
    )
    use_cache: bool = Field(
        default=True,
        description="Whether to use cached model",
        example=True
    )


class HealthResponse(BaseModel):
    """Response model for health check."""
    
    status: str
    timestamp: str
    supabase_connected: bool


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_model_from_supabase(model_name: str, bucket_prefix: str, use_cache: bool = True) -> Any:
    """
    Load model from Supabase storage (with caching).
    
    Args:
        model_name: Model filename (e.g., 'model_fkrtl.pkl', 'models_kasus_per_service.pkl')
        bucket_prefix: Bucket folder ('faskes', 'penyakit', 'peserta')
        use_cache: Whether to use cached model
    
    Returns:
        Loaded model object
    """
    cache_key = f"{bucket_prefix}_{model_name}"
    
    # Check cache first
    if use_cache and cache_key in MODEL_CACHE:
        print(f"✓ Using cached model: {cache_key}")
        return MODEL_CACHE[cache_key]
    
    # Download from Supabase
    print(f"📥 Downloading model from Supabase: {bucket_prefix}/{model_name}")
    
    try:
        storage = SupabaseModelStorage()
        bucket_name = os.getenv('SUPABASE_BUCKET_NAME', 'bucket')
        local_path = TEMP_DIR / bucket_prefix / model_name
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        model = storage.download_model(
            f"{bucket_prefix}/{model_name}",
            bucket_name=bucket_name,
            local_path=str(local_path)
        )
        
        # Cache the model
        MODEL_CACHE[cache_key] = model
        
        print(f"✓ Model loaded successfully: {cache_key}")
        return model
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model from Supabase: {str(e)}"
        )


def generate_faskes_predictions(
    model_package: Dict[str, Any],
    start_year: int,
    n_years: int,
    provinces: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Generate faskes predictions using loaded model.
    
    Args:
        model_package: Model package from Supabase
        start_year: Starting year
        n_years: Number of years to predict
        provinces: Optional list of provinces to filter
    
    Returns:
        DataFrame with predictions
    """
    # Import prediction function
    sys.path.append(str(Path(__file__).parent / 'faskes'))
    from predict import predict_xgboost_multistep
    
    # Get database URL
    db_url = os.getenv('DATABASE_URL')
    
    # Generate predictions with database URL
    df_pred = predict_xgboost_multistep(model_package, start_year, n_years, db_url=db_url)
    
    # Filter provinces if specified
    if provinces:
        entity_col = model_package['entity_col']
        df_pred = df_pred[df_pred[entity_col].isin(provinces)]
    
    return df_pred


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", tags=["General"])
async def root():
    """Root endpoint."""
    return {
        "message": "Healthkathon Universal Prediction API",
        "version": "2.0.0",
        "categories": ["faskes", "penyakit", "peserta"],
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint."""
    
    # Test Supabase connection
    supabase_ok = False
    try:
        storage = SupabaseModelStorage()
        storage.list_models()
        supabase_ok = True
    except Exception as e:
        print(f"Supabase health check failed: {e}")
    
    return {
        "status": "healthy" if supabase_ok else "degraded",
        "timestamp": datetime.now().isoformat(),
        "supabase_connected": supabase_ok
    }


@app.get("/models", tags=["Models"])
async def list_models():
    """List available models in Supabase storage."""
    
    try:
        storage = SupabaseModelStorage()
        models = storage.list_models('faskes/')
        
        return {
            "success": True,
            "models": [
                {
                    "name": m.get('name', ''),
                    "size": m.get('metadata', {}).get('size', 0),
                    "updated": m.get('updated_at', '')
                }
                for m in models
            ],
            "count": len(models)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(request: PredictionRequest):
    """
    Generate healthcare facility predictions.
    
    This endpoint:
    1. Downloads the requested model from Supabase (or uses cache)
    2. Generates predictions for specified years
    3. Returns predictions as JSON
    
    Example request:
    ```json
    {
        "dataset": "fkrtl",
        "start_year": 2019,
        "n_years": 3,
        "provinces": ["DKI Jakarta", "Jawa Barat"],
        "use_cache": true
    }
    ```
    """
    
    # Validate dataset
    valid_datasets = ['fkrtl', 'klinik_pratama', 'praktek_dokter']
    if request.dataset not in valid_datasets:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid dataset. Must be one of: {valid_datasets}"
        )
    
    try:
        # Load model
        model_name = f"model_{request.dataset}.pkl"
        model_package = load_model_from_supabase(model_name, 'faskes', request.use_cache)
        
        # Generate predictions
        df_pred = generate_faskes_predictions(
            model_package,
            request.start_year,
            request.n_years,
            request.provinces
        )
        
        # Extract column names
        entity_col = model_package['entity_col']
        time_col = model_package['time_col']
        target_col = model_package['target_col']
        
        # Convert to list of dicts
        predictions = df_pred[[entity_col, time_col, target_col]].to_dict('records')
        
        # Calculate totals
        total_facilities = int(df_pred[target_col].sum())
        
        # Build response
        return PredictionResponse(
            success=True,
            dataset=request.dataset,
            start_year=request.start_year,
            end_year=request.start_year + request.n_years - 1,
            predictions=predictions,
            total_facilities=total_facilities,
            model_info={
                "entity_col": entity_col,
                "time_col": time_col,
                "target_col": target_col,
                "features_used": len(model_package['features']),
                "provinces_count": df_pred[entity_col].nunique(),
                "years_predicted": request.n_years
            },
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/all", tags=["Predictions"])
async def predict_all(
    start_year: int = Field(..., ge=2019, le=2030),
    n_years: int = Field(1, ge=1, le=10),
    provinces: Optional[List[str]] = None,
    use_cache: bool = True
):
    """
    Predict all three datasets (fkrtl, klinik_pratama, praktek_dokter) at once.
    
    Returns predictions for all healthcare facility types.
    """
    
    datasets = ['fkrtl', 'klinik_pratama', 'praktek_dokter']
    results = {}
    
    for dataset in datasets:
        try:
            # Load model
            model_name = f"model_{dataset}.pkl"
            model_package = load_model_from_supabase(model_name, 'faskes', use_cache)
            
            # Generate predictions
            df_pred = generate_faskes_predictions(
                model_package,
                start_year,
                n_years,
                provinces
            )
            
            # Store results
            entity_col = model_package['entity_col']
            target_col = model_package['target_col']
            
            results[dataset] = {
                "predictions": df_pred.to_dict('records'),
                "total": int(df_pred[target_col].sum()),
                "provinces": df_pred[entity_col].nunique()
            }
            
        except Exception as e:
            results[dataset] = {
                "error": str(e)
            }
    
    return {
        "success": True,
        "start_year": start_year,
        "end_year": start_year + n_years - 1,
        "results": results,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict/penyakit", tags=["Predictions"])
async def predict_penyakit(request: PenyakitRequest):
    """
    Generate disease prediction forecasts.
    
    Predicts disease cases and participant counts per service type (RITL, RITP, RJTL, RJTP).
    
    Example request:
    ```json
    {
        "service_type": "RITL",
        "top_n": 10,
        "months": 12,
        "use_cache": true
    }
    ```
    """
    
    try:
        # Load penyakit models
        models_kasus = load_model_from_supabase('models_kasus_per_service.pkl', 'penyakit', request.use_cache)
        models_peserta = load_model_from_supabase('models_peserta_per_service.pkl', 'penyakit', request.use_cache)
        feature_names = load_model_from_supabase('feature_names.pkl', 'penyakit', request.use_cache)
        categorical_features = load_model_from_supabase('categorical_features.pkl', 'penyakit', request.use_cache)
        
        # Import prediction function
        sys.path.append(str(Path(__file__).parent / 'penyakit'))
        from predict_new import predict_penyakit_service
        
        # Get database URL
        db_url = os.getenv('DATABASE_URL')
        
        # Calculate year from months (assuming current year as base)
        year_to_predict = datetime.now().year
        
        # Generate predictions using database
        df_predictions = predict_penyakit_service(
            models_kasus=models_kasus,
            models_peserta=models_peserta,
            feature_names=feature_names,
            categorical_features=categorical_features,
            year=year_to_predict,
            service_type=request.service_type,
            db_url=db_url
        )
        
        # Filter top N per service
        results = {}
        
        if request.service_type:
            services = [request.service_type]
        else:
            services = df_predictions['service'].unique().tolist()
        
        for service in services:
            service_data = df_predictions[df_predictions['service'] == service].head(request.top_n)
            
            disease_predictions = []
            for _, row in service_data.iterrows():
                disease_predictions.append({
                    'disease': row['diagnosis'],
                    'kode_icd': row['kode_icd'],
                    'predicted_kasus': row['predicted_kasus'],
                    'predicted_peserta': row['predicted_peserta']
                })
            
            results[service] = disease_predictions
        
        return {
            "success": True,
            "category": "penyakit",
            "service_type": request.service_type or "all",
            "year": year_to_predict,
            "top_n": request.top_n,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/peserta", tags=["Predictions"])
async def predict_peserta(request: PesertaRequest):
    """
    Generate participant prediction forecasts.
    
    Predicts participant counts by segment type (geo, kelas, segmen).
    
    Example request:
    ```json
    {
        "segment_type": "geo",
        "months": 12,
        "use_cache": true
    }
    ```
    """
    
    # Validate segment type
    valid_segments = ['geo', 'kelas', 'segmen']
    if request.segment_type not in valid_segments:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid segment_type. Must be one of: {valid_segments}"
        )
    
    try:
        # Load peserta model
        model_name = f"model_{request.segment_type}.pkl"
        model_package = load_model_from_supabase(model_name, 'peserta', request.use_cache)
        
        # Import prediction function
        sys.path.append(str(Path(__file__).parent / 'peserta'))
        from predict import predict_xgboost_multistep
        
        # Get database URL
        db_url = os.getenv('DATABASE_URL')
        
        # Calculate years from months (assuming we predict future years)
        current_year = datetime.now().year
        n_years = max(1, request.months // 12)
        
        # Generate predictions with database URL
        df_pred = predict_xgboost_multistep(model_package, current_year, n_years, db_url=db_url)
        
        # Extract predictions
        entity_col = model_package['entity_col']
        time_col = model_package['time_col']
        target_col = model_package['target_col']
        
        # Convert to list of dicts
        predictions = df_pred[[entity_col, time_col, target_col]].to_dict('records')
        
        # Calculate summary statistics
        pred_values = df_pred[target_col].values
        
        return {
            "success": True,
            "category": "peserta",
            "segment_type": request.segment_type,
            "years": n_years,
            "predictions": predictions,
            "summary": {
                "total_years": n_years,
                "avg_predicted": int(np.mean(pred_values)),
                "min_predicted": int(np.min(pred_values)),
                "max_predicted": int(np.max(pred_values)),
                "total_predicted": int(np.sum(pred_values))
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.delete("/cache", tags=["Models"])
async def clear_cache():
    """Clear the model cache (force fresh downloads)."""
    
    MODEL_CACHE.clear()
    
    # Clean temp directory
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
        TEMP_DIR.mkdir()
    
    return {
        "success": True,
        "message": "Model cache cleared",
        "timestamp": datetime.now().isoformat()
    }


# ============================================================================
# STARTUP/SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    print(f"\n{'='*70}")
    print("UNIVERSAL PREDICTION API - STARTING")
    print(f"{'='*70}")
    print(f"Categories: Faskes, Penyakit, Peserta")
    print(f"Environment: {os.getenv('SUPABASE_URL', 'Not configured')}")
    print(f"Bucket: {os.getenv('SUPABASE_BUCKET_NAME', 'healthkathon-models')}")
    print(f"Temp dir: {TEMP_DIR}")
    print(f"{'='*70}\n")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("\nShutting down API...")
    
    # Clean temp files
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    
    print("✓ Cleanup complete")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv('API_PORT', 8000))
    host = os.getenv('API_HOST', '0.0.0.0')
    
    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
