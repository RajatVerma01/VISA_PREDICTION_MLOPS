from USvisa.pipeline.training_pipeline import TrainPipeline
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="evidently")

load_dotenv()  # loads MONGODB_CONNECTION_STRING (and any other vars) from .env


obj = TrainPipeline()
obj.run_pipeline()
