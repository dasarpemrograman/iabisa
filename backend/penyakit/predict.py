"""
PENYAKIT (DISEASE) PREDICTION MODULE
=====================================
Refactored for database integration with Supabase.

Author: Data Science Team
Date: November 2025
"""

import joblib
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List


def predict_penyakit_service(
    models_kasus: Dict[str, Any],
    models_peserta: Dict[str, Any],
    feature_names: List[str],
    categorical_features: List[str],
    year: int,
    service_type: Optional[str] = None,
    db_url: Optional[str] = None
) -> pd.DataFrame:
    """
    Generate disease prediction forecasts.
    
    Args:
        models_kasus: Dictionary of kasus models per service
        models_peserta: Dictionary of peserta models per service
        feature_names: List of feature names
        categorical_features: List of categorical feature names
        year: Year to predict
        service_type: Optional specific service type ('RITL', 'RITP', 'RJTL', 'RJTP')
        db_url: Database URL for fetching historical data
    
    Returns:
        DataFrame with predictions for all services or specified service
    """
    print("="*80)
    print(f"DISEASE FORECAST - Year {year}")
    print("="*80)
    
    # Get database URL
    if db_url is None:
        db_url = os.getenv('DATABASE_URL')
    
    if not db_url:
        raise ValueError("DATABASE_URL is required for fetching training data")
    
    # Determine services to predict
    SERVICE_TYPES = ['RITP', 'RJTP', 'RITL', 'RJTL']
    if service_type:
        if service_type not in SERVICE_TYPES:
            raise ValueError(f"Invalid service_type: {service_type}. Must be one of {SERVICE_TYPES}")
        services_to_predict = [service_type]
    else:
        services_to_predict = SERVICE_TYPES
    
    # Load historical data from database
    print(f"\n[1/3] Loading historical data from database...")
    
    sys.path.append(str(Path(__file__).parent.parent))
    from supabase_storage import SupabaseModelStorage
    
    storage = SupabaseModelStorage()
    # Fetch data with year filter (keep last 5 years for feature engineering)
    min_year = max(2014, year - 5)
    df_full = storage.fetch_training_data('penyakit', min_year=min_year)
    
    available_years = sorted(df_full['Tahun'].unique())
    max_year = max(available_years)
    
    print(f"✓ Loaded: {df_full.shape[0]} rows")
    print(f"   Available years: {available_years}")
    print(f"   Latest year: {max_year}")
    
    # Use latest year as basis for prediction
    df_basis = df_full[df_full['Tahun'] == max_year].copy()
    
    # Create prediction dataset
    print(f"\n[2/3] Creating {year} prediction dataset from {max_year} data...")
    
    df_new_year = df_basis.copy()
    df_new_year['Tahun'] = year
    
    # Combine historical data with new year for feature engineering
    df_full_with_new = pd.concat([df_full, df_new_year], ignore_index=True)
    
    print(f"✓ Created {year} dataset: {len(df_new_year)} records")
    
    # Feature engineering
    print("   Performing feature engineering...")
    
    # Log transformation
    df_full_with_new['Log_Kasus'] = np.log1p(df_full_with_new['Jumlah Kasus'])
    df_full_with_new['Log_Peserta'] = np.log1p(df_full_with_new['Jumlah Peserta'])
    
    # ICD Category
    df_full_with_new['ICD_Category'] = df_full_with_new['Kode ICD X'].str[0]
    df_full_with_new['ICD_SubCategory'] = df_full_with_new['Kode ICD X'].str[:3]
    
    # Time-series features per disease-service
    disease_service_history = df_full_with_new.groupby(['Kode ICD X', 'Jenis Service', 'Tahun']).agg({
        'Log_Kasus': 'sum',
        'Log_Peserta': 'sum'
    }).reset_index()
    
    # Lag features
    for lag in [1, 2]:
        disease_service_history[f'Log_Kasus_Lag{lag}Y'] = disease_service_history.groupby(
            ['Kode ICD X', 'Jenis Service'])['Log_Kasus'].shift(lag)
        disease_service_history[f'Log_Peserta_Lag{lag}Y'] = disease_service_history.groupby(
            ['Kode ICD X', 'Jenis Service'])['Log_Peserta'].shift(lag)
    
    # Growth rate
    disease_service_history['Log_Kasus_Growth'] = disease_service_history.groupby(
        ['Kode ICD X', 'Jenis Service'])['Log_Kasus'].diff()
    disease_service_history['Log_Peserta_Growth'] = disease_service_history.groupby(
        ['Kode ICD X', 'Jenis Service'])['Log_Peserta'].diff()
    
    disease_service_history = disease_service_history.drop(['Log_Kasus', 'Log_Peserta'], axis=1)
    
    # Merge
    df_full_with_new = df_full_with_new.merge(disease_service_history, on=['Kode ICD X', 'Jenis Service', 'Tahun'], how='left')
    
    # Category statistics
    icd_category_stats = df_full_with_new.groupby(['ICD_Category', 'Tahun']).agg({
        'Log_Kasus': ['mean', 'std'],
        'Log_Peserta': ['mean', 'std']
    }).reset_index()
    icd_category_stats.columns = ['ICD_Category', 'Tahun', 
                                   'Cat_Log_Kasus_Mean', 'Cat_Log_Kasus_Std',
                                   'Cat_Log_Peserta_Mean', 'Cat_Log_Peserta_Std']
    df_full_with_new = df_full_with_new.merge(icd_category_stats, on=['ICD_Category', 'Tahun'], how='left')
    
    # Disease frequency
    disease_freq = df_full_with_new.groupby('Kode ICD X').size().reset_index(name='Disease_Frequency')
    df_full_with_new = df_full_with_new.merge(disease_freq, on='Kode ICD X', how='left')
    
    # Time features
    df_full_with_new['Years_Since_2014'] = df_full_with_new['Tahun'] - 2014
    
    # Fill missing values
    df_full_with_new = df_full_with_new.fillna(0)
    
    # Convert categorical
    for cat_col in categorical_features:
        if cat_col in df_full_with_new.columns:
            df_full_with_new[cat_col] = df_full_with_new[cat_col].astype('category')
    
    # Filter only prediction year data
    df_predict = df_full_with_new[df_full_with_new['Tahun'] == year].copy()
    
    print(f"✓ Feature engineering completed")
    
    # Make predictions
    print(f"\n[3/3] Making predictions for {year}...")
    
    all_predictions = []
    
    for service in services_to_predict:
        df_service = df_predict[df_predict['Jenis Service'] == service].copy()
        
        if len(df_service) == 0:
            print(f"⚠ Skipping {service}: no data")
            continue
        
        if service not in models_kasus:
            print(f"⚠ Skipping {service}: no model")
            continue
        
        print(f"   {service}: {len(df_service)} diseases")
        
        # Features without 'Jenis Service'
        features_for_pred = [f for f in feature_names if f != 'Jenis Service']
        X_pred = df_service[features_for_pred]
        
        # Predict (in log space)
        kasus_pred_log = models_kasus[service].predict(X_pred)
        peserta_pred_log = models_peserta[service].predict(X_pred)
        
        # Convert back to original scale
        kasus_pred = np.expm1(kasus_pred_log)
        peserta_pred = np.expm1(peserta_pred_log)
        
        # Create results for this service
        for idx, row_idx in enumerate(df_service.index):
            all_predictions.append({
                'service': service,
                'kode_icd': df_service.loc[row_idx, 'Kode ICD X'],
                'diagnosis': df_service.loc[row_idx, 'Diagnosis Primer'],
                'predicted_kasus': int(np.maximum(kasus_pred[idx], 0)),
                'predicted_peserta': int(np.maximum(peserta_pred[idx], 0)),
                'year': year
            })
    
    print(f"\n✅ Predictions completed for {len(services_to_predict)} services")
    
    # Return as DataFrame
    result_df = pd.DataFrame(all_predictions)
    
    if len(result_df) > 0:
        # Sort by kasus descending
        result_df = result_df.sort_values('predicted_kasus', ascending=False).reset_index(drop=True)
    
    return result_df
