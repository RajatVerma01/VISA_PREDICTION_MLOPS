import mlflow
from sklearn.linear_model import LogisticRegression
import warnings

model = LogisticRegression()

# this will suppress standard python warnings, but maybe mlflow has its own
with mlflow.start_run():
    mlflow.sklearn.log_model(model, artifact_path="my_model")

with mlflow.start_run():
    mlflow.sklearn.log_model(model, name="my_model")
