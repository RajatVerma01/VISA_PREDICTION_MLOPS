import os
import sys

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from USvisa.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR
from USvisa.entity.artifact import (
    DataIngestionArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
    ClassificationMetricArtifact,
)
from USvisa.entity.config import ModelEvaluationConfig
from USvisa.entity.estimator import TargetValueMapping, USvisaModel
from USvisa.exception import USvisaException
from USvisa.logger import logging
from USvisa.utils.main_utils import load_object, write_yaml_file, read_yaml_file, drop_columns


class ModelEvaluation:
    def __init__(
        self,
        model_evaluation_config: ModelEvaluationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ):
        try:
            self.model_evaluation_config = model_evaluation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise USvisaException(e, sys)

    def _get_best_model_path(self) -> str:
        return os.path.join(
            self.model_evaluation_config.model_evaluation_dir,
            "best_model",
            "model.pkl",
        )

    def get_best_model(self):
        try:
            best_model_path = self._get_best_model_path()
            if not os.path.exists(best_model_path):
                logging.info("No previous best model found. This is the first run.")
                return None
            logging.info(f"Loading existing best model from: {best_model_path}")
            return load_object(file_path=best_model_path)
        except Exception as e:
            raise USvisaException(e, sys) from e

    def _prepare_test_data(self, test_df: pd.DataFrame):
        try:
            target_series = test_df[TARGET_COLUMN]
            input_df = test_df.drop(columns=[TARGET_COLUMN])

            input_df["company_age"] = CURRENT_YEAR - input_df["yr_of_estab"]
            drop_cols = self._schema_config["drop_columns"]
            input_df = drop_columns(df=input_df, cols=drop_cols)

            target_series = target_series.replace(TargetValueMapping()._asdict()).astype(int)
            return input_df, target_series
        except Exception as e:
            raise USvisaException(e, sys) from e

    def _get_metric_for_model(self, model: USvisaModel, test_df: pd.DataFrame) -> ClassificationMetricArtifact:
        try:
            input_df, y_true = self._prepare_test_data(test_df)
            y_pred = model.predict(input_df)

            if hasattr(model.trained_model_object, "predict_proba"):
                transformed = model.preprocessing_object.transform(input_df)
                y_prob = model.trained_model_object.predict_proba(transformed)[:, 1]
            else:
                y_prob = y_pred.astype(float)

            return ClassificationMetricArtifact(
                f1_score=f1_score(y_true, y_pred),
                precision_score=precision_score(y_true, y_pred),
                recall_score=recall_score(y_true, y_pred),
                roc_auc_score=roc_auc_score(y_true, y_prob),
            )
        except Exception as e:
            raise USvisaException(e, sys) from e

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            trained_model: USvisaModel = load_object(
                file_path=self.model_trainer_artifact.trained_model_file_path
            )

            train_model_metric = self.model_trainer_artifact.metric_artifact
            train_model_f1 = train_model_metric.f1_score

            best_model = self.get_best_model()
            best_model_path = None
            best_model_metric = None
            is_model_accepted = True
            improved_accuracy = 0.0

            if best_model is not None:
                best_model_path = self._get_best_model_path()
                best_model_metric = self._get_metric_for_model(best_model, test_df)
                best_model_f1 = best_model_metric.f1_score

                improved_accuracy = train_model_f1 - best_model_f1
                logging.info(
                    f"Trained model F1: {train_model_f1:.4f} | "
                    f"Best model F1: {best_model_f1:.4f} | "
                    f"Improvement: {improved_accuracy:.4f}"
                )

                if improved_accuracy < self.model_evaluation_config.changed_threshold_score:
                    logging.info(
                        f"New model improvement ({improved_accuracy:.4f}) is below "
                        f"threshold ({self.model_evaluation_config.changed_threshold_score}). "
                        f"Keeping existing model."
                    )
                    is_model_accepted = False
                else:
                    logging.info("New model is better. Accepting it.")
            else:
                logging.info("First run — new model accepted by default.")

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=is_model_accepted,
                improved_accuracy=improved_accuracy,
                best_model_path=best_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                train_model_metric_artifact=train_model_metric,
                best_model_metric_artifact=best_model_metric,
            )

            report = {
                "is_model_accepted": bool(is_model_accepted),
                "trained_model": {
                    "model_type": type(trained_model.trained_model_object).__name__,
                    "f1_score": round(float(train_model_metric.f1_score), 4),
                    "precision_score": round(float(train_model_metric.precision_score), 4),
                    "recall_score": round(float(train_model_metric.recall_score), 4),
                    "roc_auc_score": round(float(train_model_metric.roc_auc_score), 4),
                },
                "best_existing_model": {
                    "f1_score": round(float(best_model_metric.f1_score), 4) if best_model_metric else None,
                    "precision_score": round(float(best_model_metric.precision_score), 4) if best_model_metric else None,
                    "recall_score": round(float(best_model_metric.recall_score), 4) if best_model_metric else None,
                    "roc_auc_score": round(float(best_model_metric.roc_auc_score), 4) if best_model_metric else None,
                } if best_model_metric else None,
                "improved_accuracy": round(float(improved_accuracy), 4),
            }

            write_yaml_file(
                file_path=self.model_evaluation_config.report_file_path,
                content=report,
                replace=True,
            )
            logging.info(f"Evaluation report saved to: {self.model_evaluation_config.report_file_path}")
            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact

        except Exception as e:
            raise USvisaException(e, sys) from e
