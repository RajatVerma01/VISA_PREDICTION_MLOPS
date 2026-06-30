"""
Unit tests for pipeline classes (structure / interface tests — no MongoDB required).
"""
import pytest
from unittest.mock import patch, MagicMock

from USvisa.pipeline.training_pipeline import TrainPipeline
from USvisa.pipeline.prediction_pipeline import PredictionPipeline


class TestTrainPipelineStructure:
    """Verify TrainPipeline has all 6 required stage methods."""

    def test_has_start_data_ingestion(self):
        assert hasattr(TrainPipeline, "start_data_ingestion")

    def test_has_start_data_validation(self):
        assert hasattr(TrainPipeline, "start_data_validation")

    def test_has_start_data_transformation(self):
        assert hasattr(TrainPipeline, "start_data_transformation")

    def test_has_start_model_trainer(self):
        assert hasattr(TrainPipeline, "start_model_trainer")

    def test_has_start_model_evaluation(self):
        assert hasattr(TrainPipeline, "start_model_evaluation")

    def test_has_start_model_pusher(self):
        assert hasattr(TrainPipeline, "start_model_pusher")

    def test_has_run_pipeline(self):
        assert hasattr(TrainPipeline, "run_pipeline")

    def test_constructor_creates_all_configs(self):
        """TrainPipeline init should create configs for all 6 stages."""
        p = TrainPipeline()
        assert hasattr(p, "data_ingestion_config")
        assert hasattr(p, "data_validation_config")
        assert hasattr(p, "data_transformation_config")
        assert hasattr(p, "model_trainer_config")
        assert hasattr(p, "model_evaluation_config")
        assert hasattr(p, "model_pusher_config")


class TestPredictionPipelineNoModel:
    """PredictionPipeline must raise FileNotFoundError when no model exists."""

    def test_raises_file_not_found_when_no_model(self, tmp_path, monkeypatch):
        """
        Monkeypatch the working directory to a fresh temp folder so
        no saved_models/ or artifact/ directories exist.
        """
        monkeypatch.chdir(tmp_path)
        (tmp_path / "static").mkdir()
        (tmp_path / "templates").mkdir()
        (tmp_path / "saved_models").mkdir()

        with pytest.raises(FileNotFoundError):
            PredictionPipeline()
