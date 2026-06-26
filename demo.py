import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="evidently")

from dotenv import load_dotenv
load_dotenv()  # loads MONGODB_CONNECTION_STRING (and any other vars) from .env

from USvisa.pipeline.training_pipeline import TrainPipeline

obj = TrainPipeline()
obj.run_pipeline()