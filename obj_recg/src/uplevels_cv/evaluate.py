"""Shared, model-agnostic metrics so the families are compared apples-to-apples.

Every model reports into the same columns:
  detect_map / detect_map50  - COCO bbox mAP (pycocotools)
  pose_map / pose_oks        - COCO keypoint OKS mAP (2D pose models)
  mpjpe_mm                   - mean per-joint position error (3D pose only)
  latency_ms_p50/p95         - single-image inference latency
  params_millions, size_mb   - model footprint

mAP/OKS bodies delegate to pycocotools' COCOeval — the same scorer the COCO
leaderboard uses. 3D models (BlazePose) are down-projected to COCO-17 for a
comparable 2D OKS; their true 3D metric (MPJPE) needs 3D ground truth.
"""

from __future__ import annotations

import time


def benchmark_latency(predict_fn, *, imgsz: int = 640, device: str = "cuda",
                      warmup: int = 5, iters: int = 50) -> dict:
    """Time single-image forward passes. `predict_fn` takes one image tensor/array."""
    import numpy as np

    try:
        import torch

        dummy = torch.rand(3, imgsz, imgsz)
        if device == "cuda" and torch.cuda.is_available():
            dummy = dummy.cuda()
    except ImportError:
        dummy = np.random.rand(imgsz, imgsz, 3).astype("float32")

    for _ in range(warmup):
        predict_fn(dummy)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        predict_fn(dummy)
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return {
        "latency_ms_p50": round(times[len(times) // 2], 2),
        "latency_ms_p95": round(times[int(len(times) * 0.95)], 2),
    }


def model_footprint(model) -> dict:
    """Parameter count (millions) and on-disk size (MB).

    Best-effort: models that aren't a torch ``nn.Module`` (MediaPipe task bundle,
    some MMPose wrappers) return an empty dict rather than raising.
    """
    import io

    try:
        import torch

        from uplevels_cv.models import num_parameters

        torch_model = getattr(model, "model", model)
        buf = io.BytesIO()
        torch.save(torch_model.state_dict(), buf)
        return {
            "params_millions": round(num_parameters(model) / 1e6, 2),
            "size_mb": round(buf.getbuffer().nbytes / 1e6, 2),
        }
    except Exception:  # noqa: BLE001 — footprint is a nice-to-have, never fatal
        return {}


def evaluate_ultralytics(model, data, spec) -> dict:
    """Validate a YOLO or RT-DETR model; map Ultralytics metrics to shared columns."""
    metrics = model_footprint(model)
    res = model.val(verbose=False)  # runs on the val split from the data yaml
    box = getattr(res, "box", None)
    if box is not None:
        metrics["detect_map"] = round(float(box.map), 4)      # mAP@[.5:.95]
        metrics["detect_map50"] = round(float(box.map50), 4)
    pose = getattr(res, "pose", None)
    if pose is not None:
        metrics["pose_map"] = round(float(pose.map), 4)
        metrics["pose_oks"] = round(float(pose.map50), 4)
    return metrics


def evaluate_torchvision(model, data, spec, *, device: str = "cuda") -> dict:
    """Score a torchvision model with pycocotools COCOeval on the val split.

    Wired to report the shared columns; the predict-and-accumulate loop over the
    val dataloader is a TODO (same dependency on labeled data as the train loop).
    """
    metrics = model_footprint(model)
    # TODO: run model over data.images_dir(val); collect COCO-format results;
    #       COCOeval(gt, dt, 'bbox') -> detect_map/detect_map50, and
    #       COCOeval(gt, dt, 'keypoints') -> pose_map/pose_oks for pose models.
    return metrics


def evaluate_mmpose(model, data, spec) -> dict:
    """Score an MMPose top-down model (RTMPose) on the val split, 2D OKS."""
    metrics = model_footprint(model)
    # TODO: run the inferencer over data.images_dir(val), collect COCO keypoint
    #       results, COCOeval(gt, dt, 'keypoints') -> pose_map/pose_oks. Add a
    #       single-image latency via benchmark_latency wrapping the inferencer.
    return metrics


def evaluate_mediapipe(model, data, spec) -> dict:
    """Score MediaPipe BlazePose: 2D OKS on the COCO-17 projection + 3D note.

    BlazePose emits 33 landmarks with depth. Down-project to COCO-17 for a
    comparable 2D OKS; the true 3D metric (MPJPE) is left None because the COCO
    labels here are 2D — supply a 3D-annotated set (Human3.6M or a golf 3D
    capture) to fill mpjpe_mm.
    """
    metrics = {"params_millions": None}  # MediaPipe isn't a torch model
    # TODO: run PoseLandmarker over the val frames; skeleton_blazepose.to_coco17
    #       on each result; COCOeval('keypoints') -> pose_map/pose_oks.
    metrics["mpjpe_mm"] = None
    return metrics
