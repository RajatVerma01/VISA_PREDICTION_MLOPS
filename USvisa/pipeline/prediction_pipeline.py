import os
import sys
import glob
from dataclasses import dataclass

import pandas as pd

from USvisa.constants import CURRENT_YEAR, SAVED_MODEL_DIR, MODEL_FILE_NAME
from USvisa.entity.estimator import USvisaModel, TargetValueMapping
from USvisa.exception import USvisaException
from USvisa.logger import logging
from USvisa.utils.main_utils import load_object


@dataclass
class USvisaInputData:
    """Holds one prediction request's raw input features."""
    continent: str
    education_of_employee: str
    has_job_experience: str
    requires_job_training: str
    no_of_employees: int
    yr_of_estab: int
    region_of_employment: str
    prevailing_wage: float
    unit_of_wage: str
    full_time_position: str

    def as_dataframe(self) -> pd.DataFrame:
        """Convert to the DataFrame shape the model expects (after feature engineering)."""
        try:
            data = self.__dict__.copy()
            # Engineer company_age the same way as during training
            data["company_age"] = CURRENT_YEAR - data.pop("yr_of_estab")
            return pd.DataFrame([data])
        except Exception as e:
            raise USvisaException(e, sys) from e


class PredictionPipeline:
    """
    Loads the best trained model for serving predictions.

    Model resolution order (most preferred first):
    1. saved_models/model.pkl  — stable path written by ModelPusher after each accepted run
    2. artifact/<latest>/model_trainer/trained_model/model.pkl  — fallback for older runs
    """

    def __init__(self):
        self.model: USvisaModel = self._load_model()

    def _load_model(self) -> USvisaModel:
        """
        Try the stable pushed path first; fall back to scanning artifact/ runs.
        Raises FileNotFoundError with a clear message if nothing is found.
        """
        try:
            # ── Primary: stable saved_models/ path ──────────────────────────
            stable_path = os.path.join(SAVED_MODEL_DIR, MODEL_FILE_NAME)
            if os.path.exists(stable_path):
                logging.info(f"Loading model from stable path: {stable_path}")
                return load_object(file_path=stable_path)

            # ── S3 Fallback (Docker): Download from AWS if not local ────────
            bucket_name = os.getenv("MODEL_BUCKET_NAME")
            if bucket_name:
                logging.info(
                    f"Local model not found. Fetching from S3 bucket: {bucket_name}")
                import boto3
                s3_key = "model-registry/model.pkl"
                os.makedirs(os.path.dirname(stable_path), exist_ok=True)
                s3_client = boto3.client("s3")
                try:
                    s3_client.download_file(bucket_name, s3_key, stable_path)
                    logging.info(
                        f"Successfully downloaded model from S3 to {stable_path}")
                    return load_object(file_path=stable_path)
                except Exception as e:
                    logging.warning(f"Failed to download from S3: {e}")

            # ── Fallback: newest artifact run ────────────────────────────────
            logging.warning(
                f"{stable_path} not found (model pusher may not have run yet). "
                "Falling back to latest artifact run."
            )
            pattern = os.path.join(
                "artifact", "*", "model_trainer", "trained_model", "model.pkl"
            )
            candidates = sorted(glob.glob(pattern))
            if candidates:
                latest = candidates[-1]
                logging.info(f"Loading model from artifact fallback: {latest}")
                return load_object(file_path=latest)

            raise FileNotFoundError(
                "No trained model found. Run demo.py (or GET /train) to train first."
            )
        except FileNotFoundError:
            raise
        except Exception as e:
            raise USvisaException(e, sys) from e

    def predict(self, input_data: USvisaInputData) -> str:
        """
        Run prediction on a single input.
        Returns 'Certified' or 'Denied'.
        """
        try:
            df = input_data.as_dataframe()
            logging.info(
                f"Running prediction on input: {df.to_dict(orient='records')}")
            raw_prediction = self.model.predict(df)
            label_index = int(raw_prediction[0])
            label = TargetValueMapping().reverse_mapping().get(label_index, "Unknown")
            logging.info(f"Prediction result: {label}")
            return label
        except Exception as e:
            raise USvisaException(e, sys) from e
