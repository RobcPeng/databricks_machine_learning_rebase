"""Training dispatch + MLflow logging + serverless-GPU entry point.

Every training notebook calls exactly one function: ``train_spec(...)``. It:
  1. runs the family-specific training loop on a serverless GPU (via @distributed),
  2. logs params + metrics + the model artifact to MLflow, and
  3. registers the model to Unity Catalog and points the @prod alias at it.

The GPU boundary is ``run_on_gpu`` — it wraps a plain function with the
serverless-GPU ``@distributed`` decorator so the compute is provisioned only for
the training call, not the whole notebook.
"""

from __future__ import annotations

import time
from typing import Callable

from uplevels_cv.config import Family, ModelSpec, Paths
from uplevels_cv.data import DataConfig


def run_on_gpu(fn: Callable, *, gpus: int = 1, gpu_type: str = "A10"):
    """Wrap `fn` to execute on serverless GPU compute.

    Uses the serverless-GPU API when available; falls back to running in-process
    (e.g. on a classic GPU cluster or locally) so the same code path works
    everywhere. See README for enabling serverless GPU.
    """
    try:
        from serverless_gpu.api import distributed

        return distributed(gpus=gpus, gpu_type=gpu_type)(fn)
    except ImportError:
        return fn


def train_spec(spec: ModelSpec, data: DataConfig, *, experiment_root: str,
               gpu_type: str = "A10", overrides: dict | None = None) -> dict:
    """Train one model spec end-to-end and return its metrics dict."""
    import mlflow
    from mlflow import MlflowClient

    hp = {**spec.hparams, **(overrides or {})}
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(f"{experiment_root}/{spec.family.value}")

    with mlflow.start_run(run_name=spec.key) as run:
        mlflow.log_params({"model_key": spec.key, "family": spec.family.value,
                           "task": spec.task.value, "weights": spec.weights, **hp})

        trainer = _yolo_train if spec.family is Family.YOLO else _torchvision_train
        gpu_trainer = run_on_gpu(trainer, gpu_type=gpu_type)
        started = time.time()
        result = gpu_trainer(spec, data, hp)  # -> (model, metrics, artifact_path)
        model, metrics, artifact_path = result
        metrics["train_seconds"] = round(time.time() - started, 1)

        mlflow.log_metrics(metrics)
        model_name = data.paths.model_name(spec.key)
        info = _log_and_register(spec, model, artifact_path, model_name)

        MlflowClient(registry_uri="databricks-uc").set_registered_model_alias(
            model_name, "prod", info.registered_model_version)
        metrics["run_id"] = run.info.run_id
        metrics["model_version"] = info.registered_model_version
        return metrics


# --- family-specific training loops (run on the GPU) -----------------------

def _yolo_train(spec: ModelSpec, data: DataConfig, hp: dict):
    """Fine-tune an Ultralytics YOLO11 model and evaluate it."""
    from uplevels_cv.data import export_yolo_dataset
    from uplevels_cv.evaluate import benchmark_latency, evaluate_yolo
    from uplevels_cv.models import build_model

    model = build_model(spec)
    data_yaml = export_yolo_dataset(data, keypoints=(spec.task.value == "pose"))
    model.train(data=data_yaml, epochs=hp["epochs"], imgsz=hp["imgsz"],
                batch=hp["batch"], lr0=hp["lr"], project=data.paths.volume_root + "/artifacts")

    metrics = evaluate_yolo(model, data, spec)
    metrics.update(benchmark_latency(lambda x: model.predict(x, verbose=False),
                                     imgsz=hp["imgsz"]))
    best_pt = model.trainer.best if getattr(model, "trainer", None) else spec.weights
    return model, metrics, str(best_pt)


def _torchvision_train(spec: ModelSpec, data: DataConfig, hp: dict):
    """Fine-tune a torchvision detection/keypoint model.

    The train loop itself (dataloaders over the COCO json, optimizer, epochs) is
    the one substantial TODO — it depends on your labeled data. Everything around
    it (build, evaluate, latency, MLflow, registration) is wired.
    """
    import torch

    from uplevels_cv.evaluate import benchmark_latency, evaluate_torchvision
    from uplevels_cv.models import build_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(spec).to(device)

    # TODO: standard torchvision detection loop — build COCO dataloaders from
    #       data.coco_json(...), SGD/AdamW, hp["epochs"] passes. See
    #       torchvision references/detection for the canonical training script.

    metrics = evaluate_torchvision(model, data, spec, device=device)
    metrics.update(benchmark_latency(model, imgsz=hp["imgsz"], device=device))
    return model, metrics, None  # artifact logged via mlflow.pytorch below


def _log_and_register(spec: ModelSpec, model, artifact_path: str | None, model_name: str):
    """Log the model artifact to MLflow and register it to UC."""
    import mlflow

    if spec.family is Family.YOLO and artifact_path:
        # Log the trained .pt file; wrap in a pyfunc if you need serving.
        mlflow.log_artifact(artifact_path, artifact_path="weights")
        return mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/weights",
                                     model_name)
    return mlflow.pytorch.log_model(model, name="model", registered_model_name=model_name)
