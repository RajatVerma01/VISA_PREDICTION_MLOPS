"""
US Visa Approval Predictor — FastAPI Application
=================================================
Routes:
  GET  /              → HTML prediction form
  POST /predict       → Form submission → HTML result
  GET  /health        → Liveness probe (JSON)
  GET  /train         → Trigger training pipeline (HTML)
  GET  /monitor       → Evidently data drift report (HTML)
  POST /api/v1/predict → JSON prediction API (with Pydantic validation)
"""
import os
import sys
import glob
import json
from datetime import date
from typing import Literal, Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from USvisa.pipeline.prediction_pipeline import PredictionPipeline, USvisaInputData
from USvisa.pipeline.training_pipeline import TrainPipeline
from USvisa.constants import SAVED_MODEL_DIR, MODEL_FILE_NAME
from USvisa.exception import USvisaException
from USvisa.logger import logging

# ── App setup ────────────────────────────────────────────────────
app = FastAPI(
    title="US Visa Approval Predictor",
    description="Predicts whether a US visa application will be Certified or Denied.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Pydantic schema for JSON API ─────────────────────────────────
CURRENT_YEAR = date.today().year

class PredictRequest(BaseModel):
    """Input schema for the JSON prediction API endpoint."""
    continent: Literal["Asia", "Europe", "North America", "South America", "Africa", "Oceania"] = Field(
        ..., example="Asia"
    )
    education_of_employee: Literal["High School", "Bachelor's", "Master's", "Doctorate"] = Field(
        ..., example="Master's"
    )
    has_job_experience: Literal["Y", "N"] = Field(..., example="Y")
    requires_job_training: Literal["Y", "N"] = Field(..., example="N")
    no_of_employees: int = Field(..., ge=1, example=500)
    yr_of_estab: int = Field(..., ge=1800, le=CURRENT_YEAR, example=1995)
    region_of_employment: Literal["Northeast", "South", "West", "Midwest", "Island"] = Field(
        ..., example="Northeast"
    )
    prevailing_wage: float = Field(..., gt=0, example=75000.0)
    unit_of_wage: Literal["Hour", "Week", "Month", "Year"] = Field(..., example="Year")
    full_time_position: Literal["Y", "N"] = Field(..., example="Y")

    @field_validator("yr_of_estab")
    @classmethod
    def year_not_in_future(cls, v: int) -> int:
        if v > CURRENT_YEAR:
            raise ValueError(f"yr_of_estab cannot be in the future (max {CURRENT_YEAR})")
        return v

    @field_validator("prevailing_wage")
    @classmethod
    def wage_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("prevailing_wage must be greater than 0")
        return v


class PredictResponse(BaseModel):
    prediction: str
    confidence: str
    model_loaded_from: str


# ── Helper ───────────────────────────────────────────────────────
def _model_path() -> str:
    stable = os.path.join(SAVED_MODEL_DIR, MODEL_FILE_NAME)
    if os.path.exists(stable):
        return stable
    candidates = sorted(glob.glob(
        os.path.join("artifact", "*", "model_trainer", "trained_model", "model.pkl")
    ))
    return candidates[-1] if candidates else ""


# ────────────────────────────────────────────────────────────────
# HTML Routes
# ────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def home(request: Request):
    """Render the main prediction form."""
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": None, "error": None}
    )


@app.post("/predict", response_class=HTMLResponse, tags=["UI"])
async def predict_form(
    request: Request,
    continent: str = Form(...),
    education_of_employee: str = Form(...),
    has_job_experience: str = Form(...),
    requires_job_training: str = Form(...),
    no_of_employees: int = Form(...),
    yr_of_estab: int = Form(...),
    region_of_employment: str = Form(...),
    prevailing_wage: float = Form(...),
    unit_of_wage: str = Form(...),
    full_time_position: str = Form(...),
):
    """Handle form submission and return prediction result."""
    try:
        input_data = USvisaInputData(
            continent=continent,
            education_of_employee=education_of_employee,
            has_job_experience=has_job_experience,
            requires_job_training=requires_job_training,
            no_of_employees=no_of_employees,
            yr_of_estab=yr_of_estab,
            region_of_employment=region_of_employment,
            prevailing_wage=prevailing_wage,
            unit_of_wage=unit_of_wage,
            full_time_position=full_time_position,
        )
        pipeline = PredictionPipeline()
        result = pipeline.predict(input_data)
        logging.info(f"Prediction served via form: {result}")
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "result": result, "error": None}
        )
    except FileNotFoundError as e:
        logging.error(str(e))
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request, "result": None,
                "error": "No trained model found. Please run the training pipeline first (GET /train)."
            }
        )
    except Exception as e:
        logging.error(f"Form prediction error: {e}")
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "result": None, "error": f"Prediction failed: {e}"}
        )


@app.get("/train", response_class=HTMLResponse, tags=["UI"])
async def train(request: Request):
    """Trigger the full 6-stage training pipeline."""
    try:
        logging.info("Training pipeline triggered via /train endpoint")
        pipeline = TrainPipeline()
        pipeline.run_pipeline()
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request, "result": None, "error": None,
                "train_msg": "✅ Training pipeline completed successfully!"
            }
        )
    except Exception as e:
        logging.error(f"Training error: {e}")
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request, "result": None,
                "error": f"Training failed: {str(e)}", "train_msg": None
            }
        )


@app.get("/monitor", response_class=HTMLResponse, tags=["Observability"])
async def monitor(request: Request):
    """
    Generate and return a live Evidently data drift report.
    Compares the latest training data (reference) vs. test data (current).
    """
    try:
        import pandas as pd
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset

        # Find the most recent train/test CSVs from artifact runs
        train_csvs = sorted(glob.glob(
            os.path.join("artifact", "*", "data_ingestion", "ingested", "train.csv")
        ))
        test_csvs = sorted(glob.glob(
            os.path.join("artifact", "*", "data_ingestion", "ingested", "test.csv")
        ))

        if not train_csvs or not test_csvs:
            return HTMLResponse(
                content="<h2>No pipeline data found. Run the training pipeline first (GET /train).</h2>",
                status_code=200
            )

        reference_df = pd.read_csv(train_csvs[-1])
        current_df = pd.read_csv(test_csvs[-1])

        # Drop target column for drift analysis
        for df in [reference_df, current_df]:
            if "case_status" in df.columns:
                df.drop(columns=["case_status"], inplace=True)

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference_df, current_data=current_df)

        html_content = report.get_html()
        logging.info("Drift monitoring report generated successfully")
        return HTMLResponse(content=html_content)

    except Exception as e:
        logging.error(f"Monitor error: {e}")
        return HTMLResponse(
            content=f"<h2>Monitor error: {e}</h2><p>Ensure the training pipeline has been run at least once.</p>",
            status_code=500
        )


# ────────────────────────────────────────────────────────────────
# System Routes
# ────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health():
    """
    Liveness probe for load balancers, Kubernetes, and Docker HEALTHCHECK.
    Returns 200 OK with model status.
    """
    model_path = _model_path()
    model_status = "loaded" if model_path else "not_found"
    return JSONResponse(
        status_code=200,
        content={
            "status": "ok",
            "model": model_status,
            "model_path": model_path or "No model trained yet",
        }
    )


# ────────────────────────────────────────────────────────────────
# JSON API — v1
# ────────────────────────────────────────────────────────────────

@app.post("/api/v1/predict", response_model=PredictResponse, tags=["API v1"])
async def predict_api(payload: PredictRequest):
    """
    JSON prediction endpoint with Pydantic input validation.
    All fields are type-checked and constrained before hitting the model.
    """
    try:
        input_data = USvisaInputData(
            continent=payload.continent,
            education_of_employee=payload.education_of_employee,
            has_job_experience=payload.has_job_experience,
            requires_job_training=payload.requires_job_training,
            no_of_employees=payload.no_of_employees,
            yr_of_estab=payload.yr_of_estab,
            region_of_employment=payload.region_of_employment,
            prevailing_wage=payload.prevailing_wage,
            unit_of_wage=payload.unit_of_wage,
            full_time_position=payload.full_time_position,
        )
        pipeline = PredictionPipeline()
        result = pipeline.predict(input_data)
        model_path = _model_path()
        logging.info(f"JSON API prediction served: {result}")
        return PredictResponse(
            prediction=result,
            confidence="high" if result == "Certified" else "medium",
            model_loaded_from=model_path,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logging.error(f"API prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
