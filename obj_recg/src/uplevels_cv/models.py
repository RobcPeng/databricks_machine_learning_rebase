"""Model factory — turn a ModelSpec into a concrete model object.

Imports of torch/torchvision/ultralytics are deliberately lazy (inside the
functions) so that importing this package — for tests, for the compare step —
does not require the heavy CV stack.
"""

from __future__ import annotations

from uplevels_cv.config import Family, ModelSpec, Task
from uplevels_cv.skeleton import NUM_KEYPOINTS

# Non-background classes we detect. Background is class 0 in torchvision R-CNN.
DETECT_CLASSES = ["golfer"]  # TODO: extend with "club_head", "ball" when labeled


def build_model(spec: ModelSpec, num_classes: int = len(DETECT_CLASSES)):
    """Return a ready-to-train model for `spec`.

    - YOLO   -> an ``ultralytics.YOLO`` wrapping the pretrained checkpoint.
    - RESNET -> a torchvision detection/keypoint model with the head resized.
    - CNN    -> a lightweight torchvision one-stage detector.
    """
    if spec.family is Family.YOLO:
        from ultralytics import YOLO

        return YOLO(spec.weights)  # .train() is driven in train.py

    # torchvision families (resnet, cnn) share the same builder path.
    return _build_torchvision(spec, num_classes)


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
    if hasattr(model.roi_heads, "box_predictor"):          # Faster R-CNN
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, n)
    else:
        # SSDlite / RetinaNet expose a classification head instead of roi_heads;
        # the pretrained head is fine for a baseline. TODO: swap head for `n`
        # classes if you retrain from the COCO-pretrained weights.
        pass
    return model


def num_parameters(model) -> int:
    """Trainable parameter count. Works for torch models; YOLO wraps a torch model."""
    torch_model = getattr(model, "model", model)
    return sum(p.numel() for p in torch_model.parameters() if p.requires_grad)
