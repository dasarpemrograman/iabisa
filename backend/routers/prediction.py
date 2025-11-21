import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# --- LOGGING SETUP ---
logger = logging.getLogger("PredictionService")
logger.setLevel(logging.INFO)

# --- OPTIONAL IMPORTS (Graceful Degradation) ---
# We try to import the real ML modules. If missing, we set flags to use mock logic.
try:

    HAS_FASKES_MODULE = True
except ImportError:
    logger.warning("Module 'faskes.predict' not found. Using MOCK prediction logic.")
    HAS_FASKES_MODULE = False

try:
    from supabase_storage import SupabaseModelStorage

    HAS_SUPABASE = True
except ImportError:
    logger.warning("Module 'supabase_storage' not found. Using local/mock storage.")
    HAS_SUPABASE = False


# --- CONFIGURATION ---
router = APIRouter(tags=["Predictions"])
TEMP_DIR = Path(tempfile.gettempdir()) / "faskes_models"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# In-memory cache for loaded models to prevent re-downloading/re-loading
# Key: str (model_name), Value: Any (The model object)
MODEL_CACHE: Dict[str, Any] = {}


# --- DATA MODELS ---


class PredictionRequest(BaseModel):
    dataset: str = Field(
        ..., description="Dataset: 'fkrtl', 'klinik_pratama', 'praktek_dokter'"
    )
    start_year: int = Field(..., ge=2019, le=2030)
    n_years: int = Field(default=1, ge=1, le=10)
    provinces: Optional[List[str]] = None
    use_cache: bool = True


class PredictionResult(BaseModel):
    success: bool
    predictions: List[Dict[str, Any]]
    total_facilities: int
    metadata: Optional[Dict[str, Any]] = None


# --- SERVICE LAYER ---
# This class handles the logic, separate from the HTTP API, making it callable from main.py


class PredictionService:
    @staticmethod
    def _get_mock_prediction(
        start_year: int, n_years: int, provinces: Optional[List[str]]
    ) -> pd.DataFrame:
        """Generates realistic dummy data for UI testing when ML backend is missing."""
        years = range(start_year, start_year + n_years + 1)
        data = []

        # Default provinces if none provided
        prov_list = (
            provinces
            if provinces
            else ["DKI JAKARTA", "JAWA BARAT", "JAWA TIMUR", "BALI"]
        )

        for prov in prov_list:
            base_val = np.random.randint(50, 200)
            growth_rate = 1.05  # 5% growth
            for i, year in enumerate(years):
                val = int(base_val * (growth_rate**i)) + np.random.randint(-5, 5)
                data.append({"province": prov, "year": year, "prediction": val})
        return pd.DataFrame(data)

    @classmethod
    def _load_model(cls, model_name: str, bucket_prefix: str, use_cache: bool) -> Any:
        cache_key = f"{bucket_prefix}/{model_name}"

        if use_cache and cache_key in MODEL_CACHE:
            return MODEL_CACHE[cache_key]

        if not HAS_SUPABASE:
            # Return a dummy model object that functions expect
            return {
                "entity_col": "province",
                "time_col": "year",
                "target_col": "prediction",
                "is_mock": True,
            }

        logger.info(f"Downloading model: {bucket_prefix}/{model_name}")
        try:
            storage = SupabaseModelStorage()
            local_path = TEMP_DIR / bucket_prefix / model_name
            local_path.parent.mkdir(parents=True, exist_ok=True)

            # Download logic
            model = storage.download_model(
                f"{bucket_prefix}/{model_name}",
                use_latest=True,
                local_path=str(local_path),
            )
            MODEL_CACHE[cache_key] = model
            return model
        except Exception as e:
            logger.error(f"Failed to load model from Supabase: {e}")
            raise HTTPException(status_code=500, detail=f"Model load failed: {str(e)}")

    @classmethod
    async def predict_faskes(
        cls,
        dataset: str,
        start_year: int,
        n_years: int,
        provinces: Optional[List[str]] = None,
    ) -> PredictionResult:
        """
        Core logic to predict facility numbers.
        Returns a Pydantic model for easy consumption by both API and Agent.
        """
        valid_datasets = ["fkrtl", "klinik_pratama", "praktek_dokter"]
        if dataset not in valid_datasets:
            # Fuzzy match or default to fkrtl if invalid
            dataset = "fkrtl"

        try:
            # 1. Load Model
            model_package = cls._load_model(
                f"model_{dataset}.pkl", "faskes", use_cache=True
            )

            # 2. Run Prediction (Real or Mock)
            if HAS_FASKES_MODULE and not model_package.get("is_mock"):
                df_pred = predict_xgboost_multistep(model_package, start_year, n_years)
            else:
                logger.info("Running Mock Prediction Logic")
                df_pred = cls._get_mock_prediction(start_year, n_years, provinces)
                # Ensure columns match what the real model would output
                model_package = {
                    "entity_col": "province",
                    "time_col": "year",
                    "target_col": "prediction",
                }

            # 3. Filter by Province
            if provinces:
                # Case-insensitive filter
                mask = (
                    df_pred[model_package["entity_col"]]
                    .astype(str)
                    .str.upper()
                    .isin([p.upper() for p in provinces])
                )
                df_pred = df_pred[mask]

            # 4. Format Output
            # Group by year to get totals for the chart if no specific province selected
            # Or return full granular data

            predictions = df_pred[
                [
                    model_package["entity_col"],
                    model_package["time_col"],
                    model_package["target_col"],
                ]
            ].to_dict("records")

            total_facilities = int(df_pred[model_package["target_col"]].sum())

            return PredictionResult(
                success=True,
                predictions=predictions,
                total_facilities=total_facilities,
                metadata={"dataset": dataset, "years": n_years},
            )

        except Exception as e:
            logger.exception("Prediction failed inside Service")
            raise RuntimeError(f"Prediction Service Error: {str(e)}")


# --- HTTP ENDPOINTS (Controller Layer) ---


@router.get("/health/predictions")
async def prediction_health_check():
    return {
        "status": "healthy",
        "modules": {
            "faskes_ml": "loaded" if HAS_FASKES_MODULE else "missing (using mock)",
            "supabase": "loaded" if HAS_SUPABASE else "missing (using mock)",
        },
    }


@router.post("/predict", response_model=PredictionResult)
async def predict_faskes_endpoint(request: PredictionRequest):
    try:
        result = await PredictionService.predict_faskes(
            dataset=request.dataset,
            start_year=request.start_year,
            n_years=request.n_years,
            provinces=request.provinces,
        )
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/cache")
async def clear_cache():
    MODEL_CACHE.clear()
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
        TEMP_DIR.mkdir()
    return {"success": True, "message": "Cache cleared"}
