import sys
import os
from typing import Tuple

import mlflow
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score

from USvisa.entity.artifact import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ClassificationMetricArtifact,
)
from USvisa.entity.config import ModelTrainerConfig
from USvisa.entity.estimator import USvisaModel
from USvisa.exception import USvisaException
from USvisa.logger import logging
from USvisa.utils.main_utils import load_numpy_array_data, load_object, save_object


class ModelTrainer:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig,
    ):
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config
        except Exception as e:
            raise USvisaException(e, sys)

    @staticmethod
    def get_classification_metric(y_true: np.array, y_pred: np.array, y_prob: np.array) -> ClassificationMetricArtifact:
        return ClassificationMetricArtifact(
            f1_score=f1_score(y_true, y_pred),
            precision_score=precision_score(y_true, y_pred),
            recall_score=recall_score(y_true, y_pred),
            roc_auc_score=roc_auc_score(y_true, y_prob),
        )

    def _get_candidates(self):
        """
        Returns models with the best hyperparameters found during notebook exploration.

        Notebook findings (test accuracy after SMOTEENN resampling):
        - KNN (n_neighbors=4, weights='distance'): 96.86%  <-- best
        """
        return {
            "KNeighbors": KNeighborsClassifier(
                n_neighbors=4,
                weights="distance",
                algorithm="auto",
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=200,
                max_features="sqrt",
                max_depth=None,
                random_state=42
            )
        }

    def find_best_model(
        self, x_train: np.array, y_train: np.array, x_test: np.array, y_test: np.array
    ) -> Tuple[object, ClassificationMetricArtifact, float]:
        try:
            candidates = self._get_candidates()
            best_model = None
            best_metric = None
            best_f1 = -1.0
            overall_best_threshold = 0.5

            # ── MLflow experiment setup ──────────────────────────────────────
            mlflow.set_tracking_uri("sqlite:///mlflow.db")
            mlflow.set_experiment("usvisa-training")

            with mlflow.start_run(run_name="model_selection"):
                mlflow.set_tag("stage", "model_trainer")

                for name, model in candidates.items():
                    logging.info(f"Training {name}...")

                    with mlflow.start_run(run_name=name, nested=True):
                        model.fit(x_train, y_train)
                        if hasattr(model, "predict_proba"):
                            y_prob = model.predict_proba(x_test)[:, 1]
                            
                            # Find optimal threshold to maximize F1
                            best_threshold_for_model = 0.5
                            best_f1_for_model = -1.0
                            
                            for thresh in [0.5]:
                                temp_pred = np.where(y_prob >= thresh, 1, 0)
                                temp_f1 = f1_score(y_test, temp_pred)
                                if temp_f1 > best_f1_for_model:
                                    best_f1_for_model = temp_f1
                                    best_threshold_for_model = thresh
                            
                            y_pred = np.where(y_prob >= best_threshold_for_model, 1, 0)
                            logging.info(f"Optimal threshold for {name} found: {best_threshold_for_model:.2f}")
                        else:
                            y_pred = model.predict(x_test)
                            y_prob = y_pred.astype(float)
                            best_threshold_for_model = 0.5

                        metric = self.get_classification_metric(y_test, y_pred, y_prob)
                        acc = accuracy_score(y_test, y_pred)

                        logging.info(
                            f"{name} — Accuracy: {acc:.4f} | F1: {metric.f1_score:.4f} | "
                            f"Precision: {metric.precision_score:.4f} | Recall: {metric.recall_score:.4f} | "
                            f"ROC-AUC: {metric.roc_auc_score:.4f}"
                        )

                        # Log to MLflow
                        try:
                            mlflow.log_param("model_type", name)
                            mlflow.log_metric("optimal_threshold", float(best_threshold_for_model))
                            mlflow.log_metric("accuracy", round(float(acc), 4))
                            mlflow.log_metric("f1_score", round(float(metric.f1_score), 4))
                            mlflow.log_metric("precision", round(float(metric.precision_score), 4))
                            mlflow.log_metric("recall", round(float(metric.recall_score), 4))
                            mlflow.log_metric("roc_auc", round(float(metric.roc_auc_score), 4))
                            mlflow.sklearn.log_model(model, name=f"{name}_model")
                        except Exception as mlflow_err:
                            logging.warning(f"MLflow logging failed for {name}: {mlflow_err}")

                        if metric.f1_score > best_f1:
                            best_f1 = metric.f1_score
                            best_model = model
                            best_metric = metric
                            overall_best_threshold = best_threshold_for_model

                # Log the winner at the parent run level
                try:
                    mlflow.log_param("best_model", type(best_model).__name__)
                    mlflow.log_metric("best_f1", round(float(best_f1), 4))
                except Exception as mlflow_err:
                    logging.warning(f"MLflow parent run logging failed: {mlflow_err}")

            logging.info(f"Best model selected: {type(best_model).__name__} with F1={best_f1:.4f} and threshold={overall_best_threshold:.2f}")
            return best_model, best_metric, overall_best_threshold

        except Exception as e:
            raise USvisaException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Loading transformed train and test arrays")
            train_arr = load_numpy_array_data(
                file_path=self.data_transformation_artifact.transformed_train_file_path
            )
            test_arr = load_numpy_array_data(
                file_path=self.data_transformation_artifact.transformed_test_file_path
            )

            x_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            x_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            best_model, metric_artifact, best_thresh = self.find_best_model(x_train, y_train, x_test, y_test)

            if metric_artifact.f1_score < self.model_trainer_config.expected_accuracy:
                raise Exception(
                    f"Best model F1 score ({metric_artifact.f1_score:.4f}) is below "
                    f"expected threshold ({self.model_trainer_config.expected_accuracy}). "
                    f"Check data quality or feature engineering."
                )

            preprocessing_obj = load_object(
                file_path=self.data_transformation_artifact.transformed_object_file_path
            )

            usvisa_model = USvisaModel(
                preprocessing_object=preprocessing_obj,
                trained_model_object=best_model,
                threshold=best_thresh
            )

            logging.info(f"Saving USvisaModel: {type(best_model).__name__}")
            save_object(self.model_trainer_config.trained_model_file_path, usvisa_model)

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise USvisaException(e, sys) from e
