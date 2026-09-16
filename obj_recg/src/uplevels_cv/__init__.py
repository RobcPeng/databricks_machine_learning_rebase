"""uplevels_cv — compare vision model families on sports object-detection + pose.

Locate the athlete/equipment (bounding boxes) and trace the limb lines of a
golf swing (keypoints -> skeleton edges), then benchmark three model families
against each other on the same data:

    yolo   - Ultralytics YOLO11 (one-stage; detect + pose)
    resnet - torchvision R-CNN, ResNet-50 FPN backbone (two-stage)
    cnn    - lightweight CNN baseline (SSDlite / RetinaNet)

The public surface is intentionally small: build a spec from the zoo, train it,
evaluate it, and aggregate the runs into a comparison table.
"""

from uplevels_cv.config import Family, ModelSpec, Task, MODEL_ZOO, specs_for

__all__ = ["Family", "Task", "ModelSpec", "MODEL_ZOO", "specs_for"]
