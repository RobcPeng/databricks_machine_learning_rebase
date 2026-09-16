"""Model factory — turn a ModelSpec into a concrete model object.

Imports of torch/torchvision/ultralytics are deliberately lazy (inside the
functions) so that importing this package — for tests, for the compare step —
does not require the heavy CV stack.
"""

from __future__ import annotations

from uplevels_cv.config import Family, ModelSpec, Task
from uplevels_cv.skeleton import NUM_KEYPOINTS

# Detection class count comes from the sport (sports.SportSpec.num_detect_classes);
# callers pass it in. Default 1 = a single athlete class. Background is class 0
# in torchvision R-CNN and is added on top of num_classes.


def build_model(spec: ModelSpec, num_classes: int = 1):
    """Return a ready-to-train (or ready-to-infer) model for `spec`.

    - YOLO      -> ``ultralytics.YOLO`` wrapping the pretrained checkpoint.
    - DETR      -> ``ultralytics.RTDETR`` (same Ultralytics API as YOLO).
    - RESNET/CNN-> torchvision detection/keypoint model, head resized.
    - MMPOSE    -> an OpenMMLab MMPose top-down pose model (RTMPose).
    - MEDIAPIPE -> a MediaPipe Tasks PoseLandmarker (pretrained, inference-only).
    """
    if spec.family is Family.YOLO:
        from ultralytics import YOLO

        return YOLO(spec.weights)  # .train() is driven in train.py
    if spec.family is Family.DETR:
        from ultralytics import RTDETR

        return RTDETR(spec.weights)  # identical .train()/.val()/.predict() API
    if spec.family is Family.MMPOSE:
        return _build_mmpose(spec)
    if spec.family is Family.MEDIAPIPE:
        return _build_mediapipe(spec)

    # torchvision families (resnet, cnn) share the same builder path.
    return _build_torchvision(spec, num_classes)


def _build_mmpose(spec: ModelSpec):
    """Build an MMPose top-down pose model (RTMPose).

    RTMPose is top-down: it needs person boxes first (reuse a detector from the
    zoo, e.g. rtdetr_detect, to supply them). The easiest entry is the high-level
    inferencer, which pairs a detector + pose model by alias.

    Install is version-sensitive — mmcv must match torch/CUDA. Prefer
    ``mim install mmengine mmcv mmdet mmpose`` over plain pip. See the notebook.
    """
    from mmpose.apis import MMPoseInferencer

    # alias e.g. "rtmpose-m" resolves config + checkpoint and bundles a detector.
    return MMPoseInferencer(pose2d=spec.weights)


def _build_mediapipe(spec: ModelSpec):
    """Build a MediaPipe Tasks PoseLandmarker (BlazePose, 33 landmarks, 3D).

    Pretrained and inference-only — there is no training step. The ``.task`` model
    bundle is downloaded once into the volume; point ``model_asset_path`` at it.
    """
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    base = mp_python.BaseOptions(model_asset_path=spec.weights)
    options = vision.PoseLandmarkerOptions(base_options=base, output_segmentation_masks=False)
    return vision.PoseLandmarker.create_from_options(options)


def _build_torchvision(spec: ModelSpec, num_classes: int):
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    # num_classes includes background for torchvision detection models.
    n = num_classes + 1
    ctor = getattr(torchvision.models.detection, spec.weights)
    model = ctor(weights="DEFAULT")

    if spec.task is Task.POSE:
        # Keypoint R-CNN: keep the 17-keypoint head, just resize the box head.
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, n)
        assert model.roi_heads.keypoint_predictor is not None
        # torchvision keypoint head already emits NUM_KEYPOINTS channels.
        _ = NUM_KEYPOINTS
        return model

    # Detection heads differ by architecture — resize whichever this model has.
    if hasattr(model.roi_heads, "box_predictor"):          # Faster / Fast R-CNN
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, n)
        # Fast R-CNN (spec.hparams["proposals"] == "external") reuses this arch
        # but replaces the RPN's proposals with Selective Search boxes fed at
        # train/eval time — see selective_search() and the proposal TODO in
        # train._torchvision_train. Without them it behaves as Faster R-CNN.
    else:
        # SSDlite / RetinaNet expose a classification head instead of roi_heads;
        # the pretrained head is fine for a baseline. TODO: swap head for `n`
        # classes if you retrain from the COCO-pretrained weights.
        pass
    return model


def selective_search(image, max_proposals: int = 1000):
    """Classic region proposals for Fast R-CNN (OpenCV Selective Search).

    Fast R-CNN was designed around *external* proposals — this is that source.
    Precompute per frame into ``<volume>/proposals/`` or call live, then feed the
    boxes to the detector instead of the RPN. Returns [x1, y1, x2, y2] boxes.

    Requires opencv-contrib (``cv2.ximgproc``); the default opencv-python-headless
    does NOT include it — swap to ``opencv-contrib-python-headless`` to use this.
    """
    import cv2

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    return [[x, y, x + w, y + h] for (x, y, w, h) in ss.process()[:max_proposals]]


def num_parameters(model) -> int:
    """Trainable parameter count. Works for torch models; YOLO wraps a torch model."""
    torch_model = getattr(model, "model", model)
    return sum(p.numel() for p in torch_model.parameters() if p.requires_grad)
