US Visa Approval Prediction - MLOps Project

This project predicts whether a US visa application will be Certified** or Denied based on applicant and employer information. The main goal was to build a proper end-to-end machine learning pipeline — not just a notebook experiment, but something that can actually run in a structured, repeatable way.



What this project does?

Given details like education of the employee, number of employees in the company, prevailing wage, continent, region of employment, etc., the model predicts the visa case status.

The pipeline covers everything from pulling data out of MongoDB to training a model — with proper logging, exception handling, and artifact management at each step.



Project Structure


visa project MLOPS

USvisa/
     components/          # Core pipeline steps
          data_ingestion.py
          data_validation.py
          data_transformation.py

      pipeline/
          training_pipeline.py   # Runs all components in order
      
      entity/
          config.py        # Config dataclasses for each component
          artifact.py      # Artifact dataclasses (what each step outputs)
          estimator.py     # Target label mapping (Certified=0, Denied=1)
      constants/
          __init__.py      # All constants in one place

      data_access/
          visa_data.py     # Connects to MongoDB and pulls data

      configuration/
          mongodb_connection.py  # MongoDB client setup

      utils/
          main_utils.py    # Helper functions (read yaml, save object, etc.)

      exception/           # Custom exception class
      logger/              # Logging setup

      config/
          schema.yaml          # Column definitions, feature lists, drop columns

      notebooks/
          MONGO.ipynb
          EDA.ipynb
          FeatureEnginnering_model_training.ipynb            # Used to push data into MongoDB initially
      
      artifact/                # Auto-created when pipeline runs (gitignored)
      logs/                    # Log files (gitignored)
      demo.py                  # Entry point to run the training pipeline
      app.py                   # Flask/FastAPI app for prediction (coming soon)
      requirements.txt
      setup.py



Pipeline Steps

1. Data Ingestion
- Connects to a MongoDB Atlas cluster
- Pulls the visa dataset from the `visa_data` collection
- Saves the raw data as a CSV in `artifact/data_ingestion/feature_store/`
- Splits it into train (80%) and test (20%) sets

2. Data Validation
- Checks if all required columns are present in both train and test sets
- Validates column count against `schema.yaml`
- Runs **data drift detection** using Evidently — compares train and test distributions
- Saves a drift report as a YAML file in `artifact/data_validation/drift_report/`

3. Data Transformation
- Creates a new feature: `company_age = current_year - yr_of_estab`
- Drops columns that aren't needed for training (`case_id`, `yr_of_estab`)
- Applies preprocessing:
  - **OneHotEncoder** for: `continent`, `unit_of_wage`, `region_of_employment`
  - **OrdinalEncoder** for: `has_job_experience`, `requires_job_training`, `full_time_position`, `education_of_employee`
  - **PowerTransformer** (Yeo-Johnson) for skewed columns: `no_of_employees`, `company_age`
  - **StandardScaler** for numerical features
- Handles class imbalance using **SMOTEENN** (combination of oversampling and cleaning)
- Saves the preprocessor object and transformed numpy arrays to `artifact/data_transformation/`



How to Run

Prerequisites
- Python
- Conda environment
- MongoDB Atlas connection string in `.env` file

Setup

1. Clone the repo and go into the project folder

2. Create and activate the conda environment:

conda activate USvisa


3. Install dependencies:

pip install -r requirements.txt


4. Create a `.env` file in the root with your MongoDB connection:

MONGODB_CONNECTION_STRING="your_mongodb_connection_string_here"


5. Run the pipeline:

python demo.py


The pipeline will create timestamped artifact folders under `artifact/` and log files under `logs/`.



Tech Stack

What Tool 

Language - Python
Data Storage  - MongoDB Atlas
Data Validation - Evidently
ML  Preprocessing - Scikit-learn, Imbalanced-learn 
Imbalance Handling  - SMOTEENN 
Environment  - Conda 
Config Management  - YAML + Python dataclasses 
Logging  - Python logging module 


Dataset

The dataset contains US visa application records with the following key columns:

| Column | Description |
|--------|-------------|
| `case_id` | Unique ID for the application |
| `continent` | Applicant's continent of origin |
| `education_of_employee` | Highest education level |
| `has_job_experience` | Y/N |
| `requires_job_training` | Y/N |
| `no_of_employees` | Company size |
| `yr_of_estab` | Year company was established |
| `region_of_employment` | US region |
| `prevailing_wage` | Offered wage |
| `unit_of_wage` | Hour / Year / Week / Month |
| `full_time_position` | Y/N |
| `case_status` | **Target** — Certified or Denied |


