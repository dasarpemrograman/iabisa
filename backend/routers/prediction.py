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
from faskes.predict import predict_xgboost_multistep as predict_faskes
from peserta.predict import predict_xgboost_multistep as predict_peserta
from penyakit.predict import predict_penyakit_service
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

                # Inject metadata for table name lookup
                if isinstance(model_package, dict):
                    if folder == "faskes":
                        model_package.setdefault('dataset_name', clean_name)
                    elif folder == "peserta":
                        model_package.setdefault('segment_type', clean_name)
                    # penyakit models use different structure, handled separately

                # 1. Full Key: "penyakit/kasus_per_service"
                full_key = f"{folder}/{clean_name}"
                cls._models[full_key] = model_package

                # 2. Short Key: "kasus_per_service" (if unique)
                if clean_name not in cls._models:
                    cls._models[clean_name] = model_package

                logger.info(f"✅ Registered: {full_key} (table: {clean_name})")

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
            # Determine which prediction module to use based on model metadata
            db_url = os.getenv('DATABASE_URL')
            
            if 'dataset_name' in model_package:
                # Faskes prediction (fkrtl, klinik_pratama, praktek_dokter)
                logger.info(f"Using faskes prediction module for dataset: {model_package['dataset_name']}")
                df_pred = predict_faskes(model_package, start_year, n_years, db_url=db_url)
            elif 'segment_type' in model_package:
                # Peserta prediction (geo, kelas, segmen)
                logger.info(f"Using peserta prediction module for segment: {model_package['segment_type']}")
                df_pred = predict_peserta(model_package, start_year, n_years, db_url=db_url)
            elif dataset in ['kasus_per_service', 'peserta_per_service', 'penyakit/kasus_per_service', 'penyakit/peserta_per_service']:
                # Penyakit prediction (different structure)
                logger.info(f"Using penyakit prediction module for dataset: {dataset}")
                raise NotImplementedError("Penyakit prediction routing not yet implemented. Use separate endpoint.")
            else:
                raise ValueError(f"Cannot determine prediction module for dataset '{dataset}'. Missing metadata.")
            
            # Log prediction summary before filtering
            entity_col = model_package.get("entity_col", "province")
            target_col = model_package.get("target_col", "prediction")
            if target_col not in df_pred.columns:
                target_col = df_pred.columns[-1]
            
            # Get time column for year-based analysis
            time_col = model_package.get("time_col", "year")
            
            logger.info(f"Prediction parameters: start_year={start_year}, n_years={n_years}, requested_provinces={provinces}")
            logger.info(f"DataFrame shape: {df_pred.shape}")
            logger.info(f"Columns: {df_pred.columns.tolist()}")
            logger.info(f"Before filtering - Total predictions: {len(df_pred)}, Sum: {df_pred[target_col].sum():,.0f}, Mean: {df_pred[target_col].mean():,.0f}")
            
            # Log year breakdown
            if time_col in df_pred.columns:
                year_summary = df_pred.groupby(time_col)[target_col].agg(['count', 'sum']).reset_index()
                logger.info(f"Year breakdown:\n{year_summary.to_string(index=False)}")
            
            if len(df_pred) <= 10:
                logger.info(f"All predicted values: {df_pred[[entity_col, time_col, target_col]].to_dict('records')}")

            # Filter by Province

            if provinces and entity_col in df_pred.columns:
                logger.info(f"Filtering by provinces: {provinces}")
                logger.info(f"Entity column: {entity_col}")
                logger.info(f"Available entities (sample): {df_pred[entity_col].unique()[:10].tolist()}")
                
                # Create multiple format variations for fuzzy matching
                province_variations = set()
                for p in provinces:
                    province_variations.add(p.upper())  # BALI
                    province_variations.add(p.lower())  # bali
                    province_variations.add(p.title())  # Bali
                    province_variations.add(p.capitalize())  # Bali
                    
                # Case-insensitive matching with multiple format support
                mask = (
                    df_pred[entity_col]
                    .astype(str)
                    .str.strip()  # Remove whitespace
                    .apply(lambda x: any(
                        x.upper() == prov.upper() or  # Exact match case-insensitive
                        x.lower() == prov.lower() or  # Lowercase match
                        x == prov  # Exact match
                        for prov in provinces
                    ))
                )
                df_pred = df_pred[mask]
                
                logger.info(f"After filtering: {len(df_pred)} rows remaining")
                
                if df_pred.empty:
                    logger.warning(f"No data found for provinces {provinces}. Available entities: {list(df_pred[entity_col].unique())[:20]}")

            # Filter by requested years only
            time_col = model_package.get("time_col", "year")
            if time_col in df_pred.columns:
                end_year = start_year + n_years - 1
                year_mask = (df_pred[time_col] >= start_year) & (df_pred[time_col] <= end_year)
                df_pred = df_pred[year_mask]
                logger.info(f"After year filtering ({start_year}-{end_year}): {len(df_pred)} rows remaining")

            # Format Output
            target_col = model_package.get("target_col", "prediction")

            if target_col not in df_pred.columns:
                target_col = df_pred.columns[-1]

            predictions = df_pred.to_dict("records")
            total_facilities = (
                int(df_pred[target_col].sum()) if not df_pred.empty else 0
            )
            
            # Detailed logging of actual prediction values
            if not df_pred.empty and len(df_pred) <= 5:
                logger.info(f"Prediction details:")
                for idx, row in df_pred.iterrows():
                    logger.info(f"  Row {idx}: {entity_col}={row.get(entity_col)}, {target_col}={row.get(target_col)}, {time_col}={row.get(time_col)}")
            
            logger.info(f"Final result: {len(predictions)} records, total={total_facilities}")

            return PredictionResult(
                success=True,
                predictions=predictions,
                total_facilities=total_facilities,
                metadata={
                    "dataset": dataset,
                    "years": n_years,
                    "source": "Supabase (Direct Access)",
                    "filtered_provinces": provinces if provinces else "all",
                    "entity_column": entity_col,
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
