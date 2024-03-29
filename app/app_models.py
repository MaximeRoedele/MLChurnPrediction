from ml_models import FCNN

import _pickle
from pathlib import Path
import torch
import pandas as pd

# Make sure to find the base directory, as this can vary in Docker containers
BASE_DIR = Path("app_models.py").parent.resolve(strict=True)

# Define paths to the desired model(s) and preprocessing pipeline
MODEL_PATH = Path(f"{BASE_DIR}/ml_models/trained_models/FCNN_churn_V0.pth")
PIPELINE_PATH = Path(
    f"{BASE_DIR}/pipelines/completed_pipelines/churn_preprocessing_V0.pkl"
)

# Load in the preprocessing pipeline using _pickle (C pickle)
with open(PIPELINE_PATH, "rb") as f:
    pipeline = _pickle.load(f)

# Load in the desired model(s) using pytorch's load
model = torch.load(MODEL_PATH)


# Define the prediction pipeline used by the API
def predict(payload):
    data = pd.DataFrame.from_dict([payload], orient="columns")
    data = pipeline.transform(data)
    data = torch.from_numpy(data.values).type(torch.float32)
    churn = int(torch.round(torch.sigmoid(model(data))).item())

    if churn == 1:
        return "Yes"
    return "No"
