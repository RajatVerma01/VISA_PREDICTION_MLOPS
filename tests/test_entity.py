"""
Unit tests for USvisa/entity/* (config dataclasses, artifact dataclasses, estimator).
"""
import pandas as pd
import pytest
from datetime import date

from USvisa.entity.estimator import TargetValueMapping, USvisaModel
from USvisa.entity.artifact import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ClassificationMetricArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
    ModelPusherArtifact,
)
from USvisa.pipeline.prediction_pipeline import USvisaInputData


CURRENT_YEAR = date.today().year


class TestTargetValueMapping:
    def test_certified_maps_to_zero(self):
        mapping = TargetValueMapping()
        assert mapping.Certified == 0

    def test_denied_maps_to_one(self):
        mapping = TargetValueMapping()
        assert mapping.Denied == 1

    def test_as_dict_returns_correct_keys(self):
        d = TargetValueMapping()._asdict()
        assert "Certified" in d
        assert "Denied" in d
        assert d["Certified"] == 0
        assert d["Denied"] == 1

    def test_reverse_mapping_correct(self):
        rev = TargetValueMapping().reverse_mapping()
        assert rev[0] == "Certified"
        assert rev[1] == "Denied"


class TestUSvisaInputData:
    def _sample_input(self):
        return USvisaInputData(
            continent="Asia",
            education_of_employee="Master's",
            has_job_experience="Y",
            requires_job_training="N",
            no_of_employees=500,
            yr_of_estab=1995,
            region_of_employment="Northeast",
            prevailing_wage=75000.0,
            unit_of_wage="Year",
            full_time_position="Y",
        )

    def test_as_dataframe_returns_dataframe(self):
        df = self._sample_input().as_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_company_age_computed(self):
        df = self._sample_input().as_dataframe()
        assert "company_age" in df.columns
        assert df["company_age"].iloc[0] == CURRENT_YEAR - 1995

    def test_yr_of_estab_dropped(self):
        """yr_of_estab must be consumed and not appear in output DataFrame."""
        df = self._sample_input().as_dataframe()
        assert "yr_of_estab" not in df.columns

    def test_expected_columns_present(self):
        df = self._sample_input().as_dataframe()
        expected_cols = {
            "continent", "education_of_employee", "has_job_experience",
            "requires_job_training", "no_of_employees", "region_of_employment",
            "prevailing_wage", "unit_of_wage", "full_time_position", "company_age"
        }
        assert expected_cols == set(df.columns)


class TestArtifactDataclasses:
    def test_data_ingestion_artifact(self):
        a = DataIngestionArtifact(
            trained_file_path="train.csv", test_file_path="test.csv")
        assert a.trained_file_path == "train.csv"

    def test_model_pusher_artifact_defaults(self):
        a = ModelPusherArtifact(saved_model_path="saved_models/model.pkl")
        assert a.bucket_name is None
        assert a.s3_model_path is None

    def test_classification_metric_artifact(self):
        m = ClassificationMetricArtifact(
            f1_score=0.95, precision_score=0.94, recall_score=0.96, roc_auc_score=0.97
        )
        assert m.f1_score == 0.95
