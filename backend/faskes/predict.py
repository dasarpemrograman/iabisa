import pandas as pd
import numpy as np
import warnings
from pathlib import Path
import joblib
import sys
from datetime import datetime
import os

warnings.filterwarnings('ignore')

# ============================================================================
# FEATURE ENGINEERING (SAME AS TRAINING)
# ============================================================================

def create_panel_features(df, entity_col, time_col='Tahun', target_col='Value'):
    """Same feature engineering as training."""
    from sklearn.linear_model import LinearRegression
    
    df = df.copy()
    df = df.sort_values([entity_col, time_col])
    
    # Get minimum year from the data
    min_year = df[time_col].min()
    
    # Time features
    df['year_numeric'] = df[time_col] - min_year
    df['year_squared'] = df['year_numeric'] ** 2
    
    # Lag features
    df['lag_1'] = df.groupby(entity_col)[target_col].shift(1)
    df['lag_2'] = df.groupby(entity_col)[target_col].shift(2)
    
    # Growth rates
    df['yoy_growth'] = df.groupby(entity_col)[target_col].pct_change()
    df['yoy_growth_lag1'] = df.groupby(entity_col)['yoy_growth'].shift(1)
    
    # Entity statistics
    entity_stats = df.groupby(entity_col)[target_col].agg(['mean', 'std']).add_suffix('_entity')
    df = df.merge(entity_stats, left_on=entity_col, right_index=True)
    
    # Rolling features
    df['rolling_mean_2y'] = df.groupby(entity_col)[target_col].rolling(2, min_periods=1).mean().reset_index(0, drop=True)
    df['rolling_std_2y'] = df.groupby(entity_col)[target_col].rolling(2, min_periods=1).std().reset_index(0, drop=True)
    
    # Entity trend
    def calc_trend(group):
        if len(group) < 2:
            return 0
        X = group['year_numeric'].values.reshape(-1, 1)
        y = group[target_col].values
        try:
            return LinearRegression().fit(X, y).coef_[0]
        except:
            return 0
    
    trends = df.groupby(entity_col).apply(calc_trend)
    df['entity_trend'] = df[entity_col].map(trends)
    
    # Categorical encoding
    df['entity_encoded'] = pd.Categorical(df[entity_col]).codes
    
    # Fill NaN
    df['lag_1'] = df['lag_1'].fillna(df['rolling_mean_2y'])
    df['lag_2'] = df['lag_2'].fillna(df['rolling_mean_2y'])
    df['yoy_growth_lag1'] = df['yoy_growth_lag1'].fillna(0)
    df['rolling_std_2y'] = df['rolling_std_2y'].fillna(0)
    
    return df

# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def predict_xgboost_multistep(model_package, start_year, n_years=1, db_url=None):
    """
    Generate multi-step forecasts using XGBoost panel model.
    Uses recursive prediction where each forecast becomes input for next.
    
    Args:
        model_package: Model package with model, features, and metadata
        start_year: Starting year for predictions
        n_years: Number of years to predict
        db_url: Database URL for fetching training data
    """
    print(f"\n{'='*80}")
    print(f"PREDICTING WITH XGBOOST - {model_package['entity_col'].upper()}")
    print(f"{'='*80}")
    
    # Extract components
    model = model_package['model']
    if model is None:
        raise ValueError("Model object is missing from package. Cannot generate predictions.")
    
    features = model_package['features']
    entity_col = model_package['entity_col']
    time_col = model_package['time_col']
    target_col = model_package['target_col']
    
    # Load historical data from database
    if db_url is None:
        db_url = os.getenv('DATABASE_URL')
    
    if not db_url:
        raise ValueError("DATABASE_URL is required for fetching training data")
    
    # Determine table name from model package
    dataset_name = model_package.get('dataset_name', 'fkrtl')
    table_name = dataset_name
    
    print(f"📥 Fetching training data from database: {table_name}")
    
    # Import storage module for database access
    sys.path.append(str(Path(__file__).parent.parent))
    from supabase_storage import SupabaseModelStorage
    
    storage = SupabaseModelStorage()
    # Fetch data with year filter to reduce transfer (keep last 10 years for feature engineering)
    min_year = max(2010, start_year - 10)
    df_historical = storage.fetch_training_data(table_name, min_year=min_year)
    
    # Get unique entities from historical data
    entities = df_historical[entity_col].unique()
    
    print(f"✓ Historical data: {df_historical[time_col].min()}-{df_historical[time_col].max()}")
    print(f"✓ Predicting: {start_year}-{start_year + n_years - 1}")
    print(f"✓ Entities: {len(entities)}")
    
    # Recursive prediction
    predictions_all = []
    df_full = df_historical.copy()
    
    for year_offset in range(n_years):
        year = start_year + year_offset
        
        print(f"\n📅 Predicting year: {year}")
        
        # Create template for all entities
        future_template = pd.DataFrame({
            entity_col: entities,
            time_col: year,
            target_col: np.nan  # Will be predicted
        })
        
        # Combine with historical data for feature engineering
        df_combined = pd.concat([df_full, future_template], ignore_index=True)
        df_features = create_panel_features(df_combined, entity_col, time_col, target_col)
        
        # Extract features for prediction year
        df_pred = df_features[df_features[time_col] == year].copy()
        
        # Handle any missing features (shouldn't happen, but safety check)
        for feat in features:
            if feat not in df_pred.columns:
                print(f"   ⚠️  Missing feature: {feat}, filling with 0")
                df_pred[feat] = 0
        
        X_pred = df_pred[features]
        
        # Predict
        predictions = model.predict(X_pred)
        predictions = np.maximum(predictions, 0)  # Ensure non-negative
        predictions = np.round(predictions).astype(int)  # Convert to integers (discrete count)
        
        # Store predictions
        df_pred[target_col] = predictions
        
        predictions_all.append(df_pred[[entity_col, time_col, target_col]])
        
        # Add predictions to full dataset for next iteration
        df_full = pd.concat([df_full, df_pred[[entity_col, time_col, target_col]]], ignore_index=True)
        
        print(f"   ✓ Predicted {len(predictions)} entities")
        print(f"   ✓ Total forecast: {predictions.sum():,.0f} facilities")
        print(f"   ✓ Mean: {predictions.mean():,.0f} | Median: {np.median(predictions):,.0f}")
    
    result_df = pd.concat(predictions_all, ignore_index=True)
    
    print(f"\n✅ Prediction complete: {len(result_df)} rows")
    
    return result_df



