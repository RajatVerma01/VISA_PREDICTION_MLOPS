import sys
import numpy as np

from pandas import DataFrame
from sklearn.pipeline import Pipeline

from USvisa.exception import USvisaException
from USvisa.logger import logging



class TargetValueMapping:
    def __init__(self):
        self.Certified: int = 1
        self.Denied: int = 0

    def _asdict(self):
        return self.__dict__

    def reverse_mapping(self):
        mapping_response = self._asdict()
        return dict(zip(mapping_response.values(), mapping_response.keys()))


class USvisaModel:
    def __init__(self, preprocessing_object: Pipeline, trained_model_object: object, threshold: float = 0.5):
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object
        self.threshold = threshold

    def predict(self, dataframe: DataFrame) -> DataFrame:
        try:
            logging.info("Transforming input features and running prediction")
            transformed_feature = self.preprocessing_object.transform(dataframe)
            
            if hasattr(self.trained_model_object, "predict_proba"):
                y_prob = self.trained_model_object.predict_proba(transformed_feature)[:, 1]
                preds = np.where(y_prob >= self.threshold, 1, 0)
                return preds
            else:
                return self.trained_model_object.predict(transformed_feature)
        except Exception as e:
            raise USvisaException(e, sys) from e

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"