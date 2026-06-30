"""
Unit tests for USvisa/utils/main_utils.py
"""
import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from USvisa.utils.main_utils import (
    read_yaml_file,
    write_yaml_file,
    drop_columns,
    save_numpy_array_data,
    load_numpy_array_data,
    save_object,
    load_object,
)


SCHEMA_PATH = os.path.join("config", "schema.yaml")


class TestReadYamlFile:
    def test_reads_schema_yaml(self):
        """schema.yaml must exist and have required keys."""
        schema = read_yaml_file(SCHEMA_PATH)
        assert isinstance(schema, dict)
        assert "columns" in schema
        assert "numerical_columns" in schema
        assert "categorical_columns" in schema

    def test_raises_on_missing_file(self):
        with pytest.raises(Exception):
            read_yaml_file("nonexistent/path/file.yaml")


class TestWriteReadYamlFile:
    def test_roundtrip(self, tmp_path):
        """Write then read should produce identical dict."""
        content = {"key": "value", "nested": {"a": 1, "b": [1, 2, 3]}}
        path = str(tmp_path / "test.yaml")
        write_yaml_file(path, content)
        result = read_yaml_file(path)
        assert result == content

    def test_replace_flag_overwrites(self, tmp_path):
        """replace=True should overwrite existing file."""
        path = str(tmp_path / "test.yaml")
        write_yaml_file(path, {"v": 1})
        write_yaml_file(path, {"v": 2}, replace=True)
        result = read_yaml_file(path)
        assert result["v"] == 2


class TestDropColumns:
    def test_drops_single_column(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        result = drop_columns(df, ["b"])
        assert "b" not in result.columns
        assert "a" in result.columns
        assert "c" in result.columns

    def test_drops_multiple_columns(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        result = drop_columns(df, ["a", "c"])
        assert list(result.columns) == ["b"]

    def test_raises_on_missing_column(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(Exception):
            drop_columns(df, ["nonexistent"])


class TestNumpyArrayIO:
    def test_save_and_load_array(self, tmp_path):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        path = str(tmp_path / "test.npy")
        save_numpy_array_data(path, arr)
        loaded = load_numpy_array_data(path)
        np.testing.assert_array_equal(arr, loaded)


class TestSaveLoadObject:
    def test_save_and_load_dict(self, tmp_path):
        obj = {"key": "value", "number": 42}
        path = str(tmp_path / "obj.pkl")
        save_object(path, obj)
        loaded = load_object(path)
        assert loaded == obj

    def test_save_and_load_list(self, tmp_path):
        obj = [1, 2, 3, "hello"]
        path = str(tmp_path / "list.pkl")
        save_object(path, obj)
        loaded = load_object(path)
        assert loaded == obj
