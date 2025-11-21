import logging
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# --- REAL IMPORTS ---
from faskes.predict import predict_xgboost_multistep
from supabase_storage import SupabaseModelStorage

# --- LOGGING SETUP ---
logger = logging.getLogger("PredictionService")
logger.setLevel(logging.INFO)

# --- CONFIGURATION ---
router = APIRouter(tags=["Predictions"])
TEMP_DIR = Path(tempfile.gettempdir()) / "agentic_bi_models"
TEMP_DIR.mkdir(parents=True, exist_ok=True)


# --- DATA MODELS ---
class PredictionRequest(BaseModel):
    dataset: str = Field(
        ..., description="Dataset key (e.g. 'fkrtl', 'penyakit/kasus_per_service')"
    )
    start_year: int = Field(..., ge=2019, le=2030)
    n_years: int = Field(default=1, ge=1, le=10)
    provinces: Optional[List[str]] = None
    refresh: bool = False


class PredictionResult(BaseModel):
    success: bool
    predictions: List[Dict[str, Any]]
    total_facilities: int
    metadata: Optional[Dict[str, Any]] = None


# --- DYNAMIC REGISTRY ---
class ModelRegistry:
    """
    Manages loading of models from Supabase Storage.
    Uses a PREDEFINED list of files to avoid folder scanning issues.
    """

    _models: Dict[str, Any] = {}
    _is_initialized: bool = False

    # HARDCODED LIST based on user provision
    PREDEFINED_MODELS = [
        # Faskes
        "faskes/model_fkrtl.pkl",
        "faskes/model_klinik_pratama.pkl",
        "faskes/model_praktek_dokter.pkl",
        # Penyakit
        "penyakit/models_kasus_per_service.pkl",
        "penyakit/models_peserta_per_service.pkl",
        # Peserta
        "peserta/model_geo.pkl",
        "peserta/model_kelas.pkl",
        "peserta/model_segmen.pkl",
    ]

    @classmethod
    async def ensure_initialized(cls, force_refresh: bool = False):
        if cls._is_initialized and not force_refresh:
            return

        logger.info("Initializing Model Registry (Direct Access Mode)...")
        storage = SupabaseModelStorage()
        bucket = "bucket"

        cls._models.clear()

        for file_path in cls.PREDEFINED_MODELS:
            try:
                filename = os.path.basename(file_path)
                folder = os.path.dirname(file_path)

                # Download
                local_path = TEMP_DIR / folder / filename

                try:
                    model_package = storage.download_model(
                        path=file_path, bucket_name=bucket, local_path=str(local_path)
                    )
                except Exception as dl_err:
                    logger.warning(f"Could not download {file_path}: {dl_err}")
                    continue

                # Registration Logic
                clean_name = cls._infer_dataset_name(filename)

                # 1. Full Key: "penyakit/kasus_per_service"
                full_key = f"{folder}/{clean_name}"
                cls._models[full_key] = model_package

                # 2. Short Key: "kasus_per_service" (if unique)
                if clean_name not in cls._models:
                    cls._models[clean_name] = model_package

                logger.info(f"✅ Registered: {full_key}")

            except Exception as e:
                logger.error(f"Error registering {file_path}: {e}")

        cls._is_initialized = True
        logger.info(f"Registry Ready. Loaded {len(cls._models)} models.")

    @staticmethod
    def _infer_dataset_name(filename: str) -> str:
        """
        Cleans filename to create a readable ID.
        Ex: 'model_fkrtl.pkl' -> 'fkrtl'
        """
        name = filename.replace(".pkl", "")
        name = re.sub(r"^models?_", "", name)
        return name

    @classmethod
    def get_model(cls, dataset_name: str) -> Optional[Any]:
        return cls._models.get(dataset_name)

    @classmethod
    def list_available_datasets(cls) -> List[str]:
        return list(cls._models.keys())


# --- SERVICE LAYER ---
class PredictionService:
    @classmethod
    async def predict_faskes(
        cls,
        dataset: str,
        start_year: int,
        n_years: int,
        provinces: Optional[List[str]] = None,
        refresh: bool = False,
    ) -> PredictionResult:
        await ModelRegistry.ensure_initialized(force_refresh=refresh)

        model_package = ModelRegistry.get_model(dataset)

        # FIX: Use explicit None check to avoid ValueError with numpy arrays
        if model_package is None:
            available = ModelRegistry.list_available_datasets()
            raise RuntimeError(f"Dataset '{dataset}' not found. Available: {available}")

        # FIX: Handle case where model_package is raw model (not dict)
        if not isinstance(model_package, dict):
            model_package = {
                "model": model_package,
                "entity_col": "province",
                "time_col": "year",
                "target_col": "prediction",
            }

        try:
            # Run Prediction
            df_pred = predict_xgboost_multistep(model_package, start_year, n_years)

            # Filter by Province
            entity_col = model_package.get("entity_col", "province")

            if provinces and entity_col in df_pred.columns:
                mask = (
                    df_pred[entity_col]
                    .astype(str)
                    .str.upper()
                    .isin([p.upper() for p in provinces])
                )
                df_pred = df_pred[mask]

            # Format Output
            target_col = model_package.get("target_col", "prediction")
            time_col = model_package.get("time_col", "year")

            if target_col not in df_pred.columns:
                target_col = df_pred.columns[-1]

            predictions = df_pred.to_dict("records")
            total_facilities = (
                int(df_pred[target_col].sum()) if not df_pred.empty else 0
            )

            return PredictionResult(
                success=True,
                predictions=predictions,
                total_facilities=total_facilities,
                metadata={
                    "dataset": dataset,
                    "years": n_years,
                    "source": "Supabase (Direct Access)",
                },
            )

        except Exception as e:
            logger.exception("Prediction Logic Failed")
            raise RuntimeError(f"Prediction Error: {str(e)}")


# --- HTTP ENDPOINTS ---
@router.get("/health/predictions")
async def prediction_health_check():
    await ModelRegistry.ensure_initialized()
    return {
        "status": "healthy",
        "available_models": ModelRegistry.list_available_datasets(),
        "bucket_used": "bucket",
    }


@router.post("/predict", response_model=PredictionResult)
async def predict_faskes_endpoint(request: PredictionRequest):
    try:
        result = await PredictionService.predict_faskes(
            dataset=request.dataset,
            start_year=request.start_year,
            n_years=request.n_years,
            provinces=request.provinces,
            refresh=request.refresh,
        )
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/cache")
async def clear_cache():
    await ModelRegistry.ensure_initialized(force_refresh=True)
    return {"success": True, "message": "Registry refreshed."}
