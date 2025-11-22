import logging
import os
import pickle
import urllib.request
from pathlib import Path
from typing import Any, List, Optional

import joblib
import pandas as pd
import psycopg
from psycopg_pool import ConnectionPool
from supabase import Client, create_client

logger = logging.getLogger(__name__)

# Global connection pool
_connection_pool: Optional[ConnectionPool] = None

class SupabaseModelStorage:
    def __init__(self):
        self.url = os.getenv("SUPABASE_URL")
        self.key = os.getenv("SUPABASE_KEY")
        self.bucket_name = os.getenv("SUPABASE_BUCKET_NAME", "bucket")
        self.db_url = os.getenv("DATABASE_URL")

        if not self.url:
            logger.warning("SUPABASE_URL is missing. Storage operations may fail.")
            self.client = None
        else:
            try:
                if self.key:
                    self.client: Client = create_client(self.url, self.key)
                else:
                    self.client = None
            except Exception as e:
                logger.error(f"Failed to initialize Supabase client: {e}")
                self.client = None
        
        # Initialize connection pool
        self._init_connection_pool()

    def list_files(self, bucket: str, path: str = "") -> List[str]:
        if not self.client:
            logger.warning("Client not initialized, cannot list files.")
            return []

        try:
            res = self.client.storage.from_(bucket).list(path)
            files = []
            for item in res:
                if item["name"] == ".emptyFolderPlaceholder":
                    continue
                full_path = f"{path}/{item['name']}" if path else item["name"]
                files.append(full_path)
            return files
        except Exception as e:
            logger.error(f"Failed to list files in {bucket}/{path}: {e}")
            return []

    def _load_from_disk(self, file_path: Path) -> Any:
        """
        Attempts to load the model using pickle, then joblib.
        """
        # 1. Try Standard Pickle
        try:
            with open(file_path, "rb") as f:
                return pickle.load(f)
        except Exception as pickle_err:
            # 2. Try Joblib (Handles compressed/numpy-heavy models)
            try:
                return joblib.load(file_path)
            except Exception as joblib_err:
                # 3. Debugging: Read file header to see if it's an error page or text file
                try:
                    with open(file_path, "rb") as f:
                        header = f.read(50)
                    logger.error(f"❌ Load Failed. Header bytes: {header}")
                    if b"<!DOCTYPE html>" in header or b"<Error>" in header:
                        logger.error(
                            "⚠️ The downloaded file appears to be an HTML/XML error page, not a model."
                        )
                except:
                    pass

                raise RuntimeError(
                    f"Could not load model. Pickle error: {pickle_err}. Joblib error: {joblib_err}"
                )

    def download_model(
        self, path: str, bucket_name: str = "bucket", local_path: str = None
    ) -> Optional[Any]:
        if not local_path:
            raise ValueError("local_path must be provided")

        target_file = Path(local_path)

        # 1. Check Local Cache
        if target_file.exists():
            try:
                return self._load_from_disk(target_file)
            except Exception as e:
                logger.warning(f"Cache corrupted for {target_file.name}: {e}")
                # If cache is bad, delete it so we can re-download
                try:
                    os.remove(target_file)
                except:
                    pass

        # 2. Download using Public URL
        if not self.url:
            raise RuntimeError("SUPABASE_URL is not set in environment variables.")

        base_url = self.url.rstrip("/")
        download_url = f"{base_url}/storage/v1/object/public/{bucket_name}/{path}"

        logger.info(f"Downloading from: {download_url}")

        try:
            target_file.parent.mkdir(parents=True, exist_ok=True)

            with urllib.request.urlopen(download_url) as response:
                if response.getcode() != 200:
                    raise RuntimeError(
                        f"HTTP Error {response.getcode()} accessing {download_url}"
                    )
                    raise RuntimeError(f"HTTP Error {response.getcode()} accessing {download_url}")

                data = response.read()

            with open(target_file, "wb+") as f:
                f.write(data)

            return self._load_from_disk(target_file)

        except Exception as e:
            msg = f"Failed to download {download_url}: {e}"
            logger.error(msg)
            # Optional: Clean up partial file
            if target_file.exists():
                try:
                    os.remove(target_file)
                except:
                    pass
                raise RuntimeError(msg)
    
    def _init_connection_pool(self):
        """Initialize database connection pool."""
        global _connection_pool
        
        if _connection_pool is None and self.db_url:
            try:
                _connection_pool = ConnectionPool(
                    self.db_url,
                    min_size=2,
                    max_size=10,
                    timeout=30
                )
                logger.info("Database connection pool initialized")
            except Exception as e:
                logger.error(f"Failed to initialize connection pool: {e}")
    
    def get_database_connection(self):
        """Get a database connection from the pool."""
        global _connection_pool
        
        if _connection_pool is None:
            self._init_connection_pool()
        
        if _connection_pool:
            return _connection_pool.connection()
        else:
            # Fallback to direct connection
            if self.db_url:
                return psycopg.connect(self.db_url)
            else:
                raise RuntimeError("DATABASE_URL is not configured")
    
    def list_models(self, prefix: str = ""):
        """List models in Supabase storage with optional prefix filter."""
        try:
            files = self.list_files(self.bucket_name, prefix)
            # Return detailed info for each file
            models = []
            for file_path in files:
                models.append({
                    "name": os.path.basename(file_path),
                    "path": file_path,
                    "metadata": {"size": 0},
                    "updated_at": ""
                })
            return models
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def fetch_training_data(self, table_name: str, min_year: Optional[int] = None) -> pd.DataFrame:
        """Fetch training data from Supabase database.
        
        Args:
            table_name: Name of the table (e.g., 'fkrtl', 'klinik_pratama', 'penyakit')
            min_year: Optional minimum year to fetch (reduces data transfer)
        
        Returns:
            DataFrame with training data (id column included, pandas handles it)
        """
        try:
            with self.get_database_connection() as conn:
                # Build query with optional year filter
                if min_year is not None:
                    # Try common year column names
                    query = f'SELECT * FROM "{table_name}" WHERE "Tahun" >= {min_year}'
                else:
                    query = f'SELECT * FROM "{table_name}"'
                
                df = pd.read_sql(query, conn)
                logger.info(f"Fetched {len(df)} rows from {table_name}")
                return df
                
        except Exception as e:
            logger.error(f"Failed to fetch training data from {table_name}: {e}")
            raise RuntimeError(f"Database query failed: {e}")


def download_faskes_model(model_name: str, local_path: str) -> Any:
    """Convenience function to download faskes model from Supabase.
    
    Args:
        model_name: Model filename (e.g., 'model_fkrtl.pkl')
        local_path: Local path to save the model
    
    Returns:
        Loaded model object
    """
    storage = SupabaseModelStorage()
    bucket_name = storage.bucket_name
    model_path = f"faskes/{model_name}"
    
    return storage.download_model(model_path, bucket_name=bucket_name, local_path=local_path)
