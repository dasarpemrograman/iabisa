import logging
import os
import pickle
import urllib.request
from pathlib import Path
from typing import Any, List, Optional

import joblib  # Make sure this is installed via pyproject.toml
from supabase import Client, create_client

logger = logging.getLogger(__name__)

class SupabaseModelStorage:
    def __init__(self):
        self.url = os.getenv("SUPABASE_URL")
        self.key = os.getenv("SUPABASE_KEY")

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
