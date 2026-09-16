"""Shared, model-agnostic metrics so the families are compared apples-to-apples.

Every model reports the same columns into the comparison table:
  detect_map / detect_map50  - COCO bbox mAP (pycocotools)
  pose_map / pose_oks        - COCO keypoint OKS mAP (pose models only)
  latency_ms_p50/p95         - single-image inference latency
  params_millions, size_mb   - model footprint

The mAP/OKS bodies delegate to pycocotools' COCOeval, which is the same scorer
the COCO leaderboard uses — the fair, standard choice for detection and pose.
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
    """Parameter count (millions) and on-disk size (MB)."""
    import io

    import torch

    from uplevels_cv.models import num_parameters

    torch_model = getattr(model, "model", model)
    buf = io.BytesIO()
    torch.save(torch_model.state_dict(), buf)
    return {
        "params_millions": round(num_parameters(model) / 1e6, 2),
        "size_mb": round(buf.getbuffer().nbytes / 1e6, 2),
    }


def evaluate_yolo(model, data, spec) -> dict:
    """Validate a YOLO model; map Ultralytics metrics onto the shared columns."""
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
