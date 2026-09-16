"""Training dispatch + MLflow logging + serverless-GPU entry point.

Every training notebook calls exactly one function: ``train_spec(...)``. It:
  1. runs the family-specific training loop on a serverless GPU (via @distributed),
     or evaluates a pretrained model when the spec is inference-only,
  2. logs params + metrics + the model artifact to MLflow, and
  3. registers the model to Unity Catalog and points the @prod alias at it
     (best-effort — a model that can't be cleanly logged still lands its metrics
     in the comparison table).

The GPU boundary is ``run_on_gpu`` — it wraps a plain function with the
serverless-GPU ``@distributed`` decorator so compute is provisioned only for the
training call, not the whole notebook.
"""

from __future__ import annotations

import time
from typing import Callable

from uplevels_cv.config import Family, ModelSpec, Paths, Task
from uplevels_cv.data import DataConfig


def run_on_gpu(fn: Callable, *, gpus: int = 1, gpu_type: str = "A10"):
    """Wrap `fn` to execute on serverless GPU compute.

    Uses the serverless-GPU API when available; falls back to running in-process
    (classic GPU cluster or locally) so the same code path works everywhere.
    """
    try:
        from serverless_gpu.api import distributed

        return distributed(gpus=gpus, gpu_type=gpu_type)(fn)
    except ImportError:
        return fn


def _select_trainer(spec: ModelSpec) -> Callable:
    if not spec.trainable:
        return _inference_only                       # e.g. MediaPipe BlazePose
    if spec.family in (Family.YOLO, Family.DETR):
        return _ultralytics_train                    # YOLO11, RT-DETR
    if spec.family is Family.MMPOSE:
        return _mmpose_train                         # RTMPose
    return _torchvision_train                        # Faster/Fast R-CNN, SSDlite


def train_spec(spec: ModelSpec, data: DataConfig, *, experiment_root: str,
               gpu_type: str = "A10", overrides: dict | None = None) -> dict:
    """Train (or evaluate) one model spec end-to-end and return its metrics.

    Steps:
      1. Merge hparams (spec defaults + per-run overrides).
      2. Point MLflow at the Unity Catalog registry and the family experiment.
      3. Open a run and log the spec + hparams as params.
      4. Select the trainer for the family; wrap it for serverless GPU unless the
         spec is inference-only.
      5. Run it -> (model, metrics, artifact_path). Record wall-clock train time.
      6. Log numeric metrics.
      7. Log the model artifact and register it to UC (best-effort); on success,
         move the @prod alias to the new version.
      8. Return the metrics dict (includes run_id and, if registered, model_version).
    """
    import mlflow
    from mlflow import MlflowClient

    # 1. hyperparameters
    hp = {**spec.hparams, **(overrides or {})}

    # 2. registry + experiment (one experiment per family)
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(f"{experiment_root}/{spec.family.value}")

    # Runs are keyed by <sport>_<model> so sports don't collide in an experiment.
    run_name = f"{data.sport}_{spec.key}"
    with mlflow.start_run(run_name=run_name) as run:
        # 3. log configuration
        mlflow.log_params({"model_key": spec.key, "family": spec.family.value,
                           "task": spec.task.value, "weights": spec.weights,
                           "trainable": spec.trainable, "sport": data.sport, **hp})

        # 4. pick + place the trainer (GPU only for actual training)
        trainer = _select_trainer(spec)
        runner = run_on_gpu(trainer, gpu_type=gpu_type) if spec.trainable else trainer

        # 5. run
        started = time.time()
        model, metrics, artifact_path = runner(spec, data, hp)
        if spec.trainable:
            metrics["train_seconds"] = round(time.time() - started, 1)

        # 6. metrics
        mlflow.log_metrics({k: v for k, v in metrics.items()
                            if isinstance(v, (int, float)) and not isinstance(v, bool)})

        # 7. log + register + alias (model name namespaced by sport)
        model_name = data.paths.model_name(f"{data.sport}_{spec.key}")
        version = _log_and_register(spec, model, artifact_path, model_name)
        if version is not None:
            MlflowClient(registry_uri="databricks-uc").set_registered_model_alias(
                model_name, "prod", version)
            metrics["model_version"] = version

        # 8. return
        metrics["run_id"] = run.info.run_id
        return metrics


# --- family-specific training loops (run on the GPU) -----------------------

def _ultralytics_train(spec: ModelSpec, data: DataConfig, hp: dict):
    """Fine-tune an Ultralytics model (YOLO11 or RT-DETR) and evaluate it.

    Steps:
      1. Build the model from the checkpoint.
      2. Export the COCO labels to the YOLO dataset layout and get its data.yaml.
      3. Train for hp["epochs"], writing artifacts under the volume.
      4. Validate (mAP / OKS) and benchmark inference latency.
      5. Return the model, metrics, and the best-checkpoint path.
    """
    from uplevels_cv.data import export_yolo_dataset
    from uplevels_cv.evaluate import benchmark_latency, evaluate_ultralytics
    from uplevels_cv.models import build_model

    model = build_model(spec)                                                   # 1
    data_yaml = export_yolo_dataset(data, keypoints=(spec.task.value == "pose"))  # 2
    model.train(data=data_yaml, epochs=hp["epochs"], imgsz=hp["imgsz"],          # 3
                batch=hp["batch"], lr0=hp["lr"], project=data.paths.volume_root + "/artifacts")

    metrics = evaluate_ultralytics(model, data, spec)                           # 4
    metrics.update(benchmark_latency(lambda x: model.predict(x, verbose=False),
                                     imgsz=hp["imgsz"]))
    best_pt = model.trainer.best if getattr(model, "trainer", None) else spec.weights  # 5
    return model, metrics, str(best_pt)


def _torchvision_train(spec: ModelSpec, data: DataConfig, hp: dict):
    """Fine-tune a torchvision detection/keypoint model.

    Steps:
      1. Select the device and build the model on it.
      2. Train over the COCO dataloaders (TODO — depends on the labeled data).
      3. Evaluate (COCOeval) and benchmark inference latency.
      4. Return the model, metrics, and None (logged via the pytorch flavor).
    """
    import torch

    from uplevels_cv.evaluate import benchmark_latency, evaluate_torchvision
    from uplevels_cv.models import build_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Keypoint R-CNN detects persons only, so its box head has one foreground
    # class (person); detection models use the sport's full class set. Passing
    # num_detect_classes to a pose model would size the box head for club/ball
    # classes that carry no keypoints.
    num_classes = 1 if spec.task is Task.POSE else data.num_detect_classes
    model = build_model(spec, num_classes).to(device)

    # TODO: standard torchvision detection loop — build COCO dataloaders from
    #       data.coco_json(...), SGD/AdamW, hp["epochs"] passes. See
    #       torchvision references/detection for the canonical training script.
    #       For Fast R-CNN (hp.get("proposals") == "external"), feed
    #       models.selective_search(frame) boxes as proposals and bypass the RPN
    #       instead of letting the model generate them.

    metrics = evaluate_torchvision(model, data, spec, device=device)
    metrics.update(benchmark_latency(model, imgsz=hp["imgsz"], device=device))
    return model, metrics, None  # artifact logged via mlflow.pytorch below


def _mmpose_train(spec: ModelSpec, data: DataConfig, hp: dict):
    """Fine-tune / evaluate an MMPose top-down model (RTMPose).

    RTMPose is top-down: it needs person boxes, supplied from a detector in the
    zoo (e.g. rtdetr_detect).

    Steps:
      1. Build the MMPose inferencer.
      2. Train via the MMEngine Runner over the COCO labels (TODO).
      3. Evaluate (2D OKS) and return the model + metrics.
    """
    from uplevels_cv.evaluate import evaluate_mmpose
    from uplevels_cv.models import build_model

    model = build_model(spec)                                  # 1
    # 2. TODO: MMEngine Runner(cfg).train() over the COCO labels in the volume.
    metrics = evaluate_mmpose(model, data, spec)               # 3
    return model, metrics, None


def _inference_only(spec: ModelSpec, data: DataConfig, hp: dict):
    """Evaluate a pretrained, non-trainable model (MediaPipe BlazePose)."""
    from uplevels_cv.evaluate import evaluate_mediapipe
    from uplevels_cv.models import build_model

    model = build_model(spec)
    metrics = evaluate_mediapipe(model, data, spec)
    return model, metrics, None


def _log_and_register(spec: ModelSpec, model, artifact_path: str | None, model_name: str):
    """Log the model to MLflow and register it to UC. Returns the version or None.

    Registration is best-effort: models that don't fit a clean MLflow flavor
    (MMPose inferencer, MediaPipe task bundle) log what they can and skip
    registration with a tag — their metrics still reach the comparison table.
    """
    import mlflow

    run_id = mlflow.active_run().info.run_id
    try:
        if artifact_path:  # Ultralytics .pt (and any file-based checkpoint)
            mlflow.log_artifact(artifact_path, artifact_path="weights")
            return mlflow.register_model(f"runs:/{run_id}/weights", model_name).version
        if spec.family is Family.RESNET or spec.family is Family.CNN:
            return mlflow.pytorch.log_model(
                model, name="model", registered_model_name=model_name).registered_model_version
    except Exception as exc:  # noqa: BLE001 — never fail the run over registration
        mlflow.set_tag("register_skipped", f"{type(exc).__name__}: {exc}")
        return None
    mlflow.set_tag("register_skipped", f"no MLflow flavor wired for {spec.family.value}")
    return None
