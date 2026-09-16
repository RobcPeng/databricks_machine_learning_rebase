"""uplevels_cv — compare vision model families on object detection + pose.

The task: locate the athlete/equipment (bounding boxes) and produce the pose
keypoints of a golf swing (keypoints -> skeleton edges). Model families, all
trained/evaluated on the same data and scored on the same metrics:

    yolo      - Ultralytics YOLO11 (one-stage; detect + pose)
    detr      - RT-DETR transformer detector (Ultralytics API)
    resnet    - torchvision R-CNN, ResNet-50 FPN (Faster / Fast R-CNN, Keypoint R-CNN)
    cnn       - torchvision SSDlite (one-stage)
    mmpose    - RTMPose top-down 2D pose (OpenMMLab)
    mediapipe - BlazePose 33-landmark 3D pose (pretrained, inference-only)

Public surface: build a spec from the zoo (config.MODEL_ZOO / specs_for), train
or evaluate it (train.train_spec), and aggregate the runs into a comparison
table (compare.write_comparison).
"""

from uplevels_cv.config import Family, ModelSpec, Task, MODEL_ZOO, specs_for

__all__ = ["Family", "Task", "ModelSpec", "MODEL_ZOO", "specs_for"]
