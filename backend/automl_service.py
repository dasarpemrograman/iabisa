import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic_ai import Agent
from sklearn.linear_model import LinearRegression

logger = logging.getLogger("AutoML")


class AutoMLService:
    """
    A Production-Ready Prediction Engine.
    1. Generates SQL with strict aliasing and quoting rules.
    2. Fetches full history (No LIMIT).
    3. Trains Linear Regression for trend extrapolation.
    """

    @staticmethod
    async def run_forecast(
        user_query: str, llm_model: Any, inspector: Any, forecast_years: int = 5
    ) -> List[Dict[str, Any]]:
        # --- STEP 1: GENERATE SQL (STRICT MODE) ---
        logger.info(
            f"🤖 AutoML: Generating SQL for history. Horizon: {forecast_years} years."
        )

        history_agent = Agent(
            llm_model,
            system_prompt=(
                f"You are an Omniscient PostgreSQL Expert. \n"
                f"SCHEMA:\n{inspector.get_schema_summary()}\n"
                "TASK: Generate the PERFECT SQL query to fetch historical time-series data for the user's request.\n"
                "STRICT RULES:\n"
                "1. **FETCH ALL DATA**: Do NOT use `LIMIT`. We need the full history for training.\n"
                "2. **SORTING**: Order by the time column ASCENDING.\n"
                '3. **QUOTING**: Postgres is CASE-SENSITIVE. You MUST double-quote ALL identifiers (e.g. "Tahun", "Jumlah_Klinik").\n'
                "4. **TEXT MATCHING**: Use `ILIKE` for text comparisons (e.g. WHERE \"Provinsi\" ILIKE '%Jawa Barat%').\n"
                "5. **ALIASING**: You MUST alias the columns exactly as follows:\n"
                "   - `time_col` (for the year/date)\n"
                "   - `entity_col` (for the grouping, e.g. province. If none, use a static string 'Total' as entity_col)\n"
                "   - `target_col` (for the metric value)\n"
                "6. **OUTPUT**: Return ONLY the raw SQL. No markdown."
            ),
        )

        sql_result = await history_agent.run(f"User Query: {user_query}")

        # Parsing the response (Handling markdown fences if present)
        sql_query = sql_result.output.strip().replace("```sql", "").replace("```", "")

        logger.info(f"📜 Executing SQL: {sql_query}")

        raw_data = inspector.execute_query(sql_query)

        # --- ERROR HANDLING ---
        if not raw_data:
            logger.warning("⚠️ Query returned empty result.")
            return []

        if isinstance(raw_data, list) and len(raw_data) > 0 and "error" in raw_data[0]:
            logger.error(f"❌ Data Fetch Failed: {raw_data}")
            raise RuntimeError(f"SQL Error: {raw_data[0]['error']}")

        df = pd.DataFrame(raw_data)
        logger.info(f"✅ Fetched {len(df)} rows. Columns: {list(df.columns)}")

        # --- STEP 2: NORMALIZE COLUMNS ---
        # Since we enforced aliasing, we can mostly trust the names.
        cols = df.columns
        time_col = "time_col" if "time_col" in cols else df.columns[0]
        target_col = "target_col" if "target_col" in cols else df.columns[-1]
        entity_col = (
            "entity_col"
            if "entity_col" in cols
            else next((c for c in cols if c not in [time_col, target_col]), None)
        )

        # --- STEP 3: TRAIN & PREDICT (Linear Regression) ---
        future_preds = []

        # Ensure entity column exists for grouping
        if not entity_col:
            df["entity_col"] = "Total"
            entity_col = "entity_col"
        else:
            df[entity_col] = df[entity_col].astype(str)

        entities = df[entity_col].unique()

        # Determine Year Range
        try:
            max_year_in_data = int(df[time_col].max())
        except:
            # Fallback if time_col is datetime
            max_year_in_data = pd.to_datetime(df[time_col]).max().year

        start_pred_year = max_year_in_data + 1
        target_years = range(start_pred_year, start_pred_year + forecast_years)

        for entity in entities:
            try:
                # Prepare Group Data
                group_df = df[df[entity_col] == entity].sort_values(by=time_col)

                if len(group_df) < 2:
                    continue

                # X = Time, y = Target
                X_train = group_df[[time_col]].values.astype(int)
                y_train = group_df[target_col].values

                # Train Linear Regression (Allows upward/downward trends)
                model = LinearRegression()
                model.fit(X_train, y_train)

                # Predict
                X_future = np.array(list(target_years)).reshape(-1, 1)
                y_pred = model.predict(X_future)

                for i, year in enumerate(target_years):
                    val = int(y_pred[i])
                    future_preds.append(
                        {
                            time_col: int(year),
                            entity_col: entity,
                            target_col: max(0, val),  # Clamp negative predictions
                            "type": "forecast",
                        }
                    )

            except Exception as e:
                logger.warning(f"Training failed for {entity}: {e}")
                continue

        # Combine History + Forecast
        df["type"] = "history"
        full_dataset = df.to_dict("records") + future_preds
        full_dataset = df.to_dict("records") + future_preds

        return full_dataset
