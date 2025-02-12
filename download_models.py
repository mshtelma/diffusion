import os

from pathlib import Path
import mlflow
from mlflow import MlflowClient
from huggingface_hub import snapshot_download


mlflow.set_tracking_uri("databricks")

mlflow_run_id = "43661d71c9c4405184a6b2e71e68dafe"
artifact_path = "v2-finetune/checkpoints/ep9-ba250-rank0.pt"

local_checkpoint_path = str(Path("/root") / artifact_path)
MlflowClient().download_artifacts(mlflow_run_id,    artifact_path, "/root/")

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

with open('/root/v2-finetune/checkpoints/v1-autoencoder.pt', 'wb') as f:
    resp = w.files.download("/Volumes/shutterstock-data/checkpoints/checkpoints/v1-autoencoder.pt")
    f.write(resp.contents.read())

cache_dir = '/tmp/hf_files'
#snapshot_download("google/t5-v1_1-xxl", cache_dir=cache_dir)
snapshot_download("google/t5-v1_1-xxl")

