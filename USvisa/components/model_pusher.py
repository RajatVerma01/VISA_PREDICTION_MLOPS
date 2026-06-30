import os
import shutil
import sys
from typing import Optional

from USvisa.entity.artifact import ModelEvaluationArtifact, ModelPusherArtifact
from USvisa.entity.config import ModelPusherConfig
from USvisa.exception import USvisaException
from USvisa.logger import logging


class ModelPusher:
    """
    Copies the accepted model to a stable local path (saved_models/model.pkl)
    and optionally uploads it to an S3 bucket if AWS credentials + bucket
    name are configured via environment variables.

    The stable local path is always written so the prediction API has a
    fixed, non-timestamped location to load from regardless of S3.
    """

    def __init__(
        self,
        model_evaluation_artifact: ModelEvaluationArtifact,
        model_pusher_config: ModelPusherConfig,
    ):
        try:
            self.model_evaluation_artifact = model_evaluation_artifact
            self.model_pusher_config = model_pusher_config
        except Exception as e:
            raise USvisaException(e, sys)

    def _push_to_local(self) -> str:
        """Copy the trained model to the stable saved_models/ directory."""
        try:
            src = self.model_evaluation_artifact.trained_model_path
            dest = self.model_pusher_config.saved_model_path
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(src, dest)
            logging.info(f"Model saved locally to: {dest}")
            return dest
        except Exception as e:
            raise USvisaException(e, sys) from e

    def _push_to_s3(self, local_model_path: str) -> Optional[str]:
        """
        Upload model to S3 if MODEL_BUCKET_NAME env var is set.
        Returns the s3:// URI on success, None if S3 is not configured.
        """
        try:
            bucket_name = self.model_pusher_config.bucket_name
            if not bucket_name:
                logging.info(
                    "MODEL_BUCKET_NAME env var not set — skipping S3 upload. "
                    "Set it to enable automatic S3 model registry."
                )
                return None

            import boto3
            from botocore.exceptions import BotoCoreError, ClientError

            s3_key = self.model_pusher_config.s3_model_key
            s3_client = boto3.client("s3")

            logging.info(f"Uploading model to s3://{bucket_name}/{s3_key}")
            s3_client.upload_file(local_model_path, bucket_name, s3_key)
            s3_uri = f"s3://{bucket_name}/{s3_key}"
            logging.info(f"Model successfully pushed to S3: {s3_uri}")
            return s3_uri

        except Exception as e:
            logging.warning(
                f"S3 upload failed ({type(e).__name__}: {e}). "
                "Model is still saved locally."
            )
            return None
        except Exception as e:
            raise USvisaException(e, sys) from e

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Entry point. Copies the model locally and optionally to S3.
        Only called when model_evaluation_artifact.is_model_accepted is True.
        """
        try:
            logging.info("Starting model pusher")

            saved_model_path = self._push_to_local()
            s3_uri = self._push_to_s3(saved_model_path)

            bucket_name = self.model_pusher_config.bucket_name or None

            model_pusher_artifact = ModelPusherArtifact(
                saved_model_path=saved_model_path,
                bucket_name=bucket_name if s3_uri else None,
                s3_model_path=s3_uri,
            )

            logging.info(f"Model pusher artifact: {model_pusher_artifact}")
            return model_pusher_artifact

        except Exception as e:
            raise USvisaException(e, sys) from e
