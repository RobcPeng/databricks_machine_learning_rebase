"""Model zoo and run configuration for the benchmark.

Adding a fourth model to compare is a one-line append to ``_ZOO`` — everything
downstream (training dispatch, evaluation, the comparison table) keys off these
specs, so nothing else needs to change.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum


class Family(str, Enum):
    """The model families under comparison."""

    YOLO = "yolo"            # Ultralytics YOLO11, one-stage anchor-free
    RESNET = "resnet"        # torchvision R-CNN, ResNet-50 FPN backbone, two-stage
    CNN = "cnn"              # lightweight one-stage CNN baseline
    DETR = "detr"            # transformer detector (RT-DETR via Ultralytics)
    MMPOSE = "mmpose"        # OpenMMLab top-down 2D pose (RTMPose)
    MEDIAPIPE = "mediapipe"  # Google BlazePose, 33-landmark 3D pose


class Task(str, Enum):
    """What the model predicts."""

    DETECT = "detect"    # bounding boxes (golfer, club head, ball)
    POSE = "pose"        # 17 person keypoints -> limb lines (the swing overlay)
    POSE3D = "pose3d"    # 3D landmarks (x, y, z) -> swing plane / rotation


@dataclass(frozen=True)
class ModelSpec:
    """One trainable/benchmarkable model configuration.

    ``weights`` is the pretrained checkpoint for YOLO (e.g. ``yolo11n-pose.pt``)
    or the torchvision constructor name for resnet/cnn (e.g.
    ``keypointrcnn_resnet50_fpn``). ``hparams`` holds the knobs the trainer reads.
    """

    key: str
    family: Family
    task: Task
    weights: str
    description: str
    hparams: dict = field(default_factory=dict)
    trainable: bool = True  # False = pretrained, evaluate-only (e.g. BlazePose)


# Sensible small-but-real defaults; override per-run via notebook parameters.
_DETECT_HP = {"epochs": 50, "imgsz": 640, "batch": 16, "lr": 1e-3}
_POSE_HP = {"epochs": 80, "imgsz": 640, "batch": 16, "lr": 1e-3}

_ZOO = [
    # --- YOLO11 (Ultralytics, one-stage) ------------------------------------
    ModelSpec("yolo_detect", Family.YOLO, Task.DETECT, "yolo11n.pt",
              "YOLO11-nano. One-stage detector; outputs bounding boxes.", dict(_DETECT_HP)),
    ModelSpec("yolo_pose", Family.YOLO, Task.POSE, "yolo11n-pose.pt",
              "YOLO11-nano pose. Outputs 17 COCO keypoints.", dict(_POSE_HP)),

    # --- ResNet-50 FPN R-CNN (torchvision, two-stage) -----------------------
    ModelSpec("resnet_detect", Family.RESNET, Task.DETECT, "fasterrcnn_resnet50_fpn_v2",
              "Faster R-CNN, ResNet-50 FPN backbone. Two-stage detector.", dict(_DETECT_HP)),
    ModelSpec("resnet_pose", Family.RESNET, Task.POSE, "keypointrcnn_resnet50_fpn",
              "Keypoint R-CNN, ResNet-50 FPN backbone. Outputs 17 COCO keypoints.", dict(_POSE_HP)),
    ModelSpec("fast_rcnn_detect", Family.RESNET, Task.DETECT, "fasterrcnn_resnet50_fpn",
              "Fast R-CNN. torchvision has no native Fast R-CNN; runs the Faster "
              "R-CNN architecture fed external Selective Search proposals with the "
              "RPN bypassed.",
              {**_DETECT_HP, "proposals": "external"}),

    # --- SSDlite (torchvision, one-stage) -----------------------------------
    ModelSpec("cnn_detect", Family.CNN, Task.DETECT, "ssdlite320_mobilenet_v3_large",
              "SSDlite, MobileNetV3-Large backbone. One-stage detector.",
              {**_DETECT_HP, "batch": 32}),

    # --- RT-DETR (Ultralytics, transformer) ---------------------------------
    ModelSpec("rtdetr_detect", Family.DETR, Task.DETECT, "rtdetr-l.pt",
              "RT-DETR (large). Transformer detector; uses the Ultralytics API.",
              dict(_DETECT_HP)),

    # --- RTMPose (MMPose, top-down 2D pose) ---------------------------------
    ModelSpec("rtmpose_pose", Family.MMPOSE, Task.POSE, "rtmpose-m",
              "RTMPose-m (MMPose). Top-down 2D pose; 17 COCO keypoints. Requires "
              "upstream person boxes from a detector.",
              dict(_POSE_HP)),

    # --- BlazePose (MediaPipe, 3D pose) — inference-only --------------------
    ModelSpec("blazepose_pose3d", Family.MEDIAPIPE, Task.POSE3D, "pose_landmarker_full.task",
              "MediaPipe BlazePose (full). 33 landmarks with 3D (x, y, z). "
              "Pretrained; evaluated, not trained.",
              {"landmarks": 33}, trainable=False),
]

MODEL_ZOO: dict[str, ModelSpec] = {s.key: s for s in _ZOO}


def specs_for(family: Family | str | None = None,
              task: Task | str | None = None) -> list[ModelSpec]:
    """Filter the zoo by family and/or task. No filter returns everything."""
    fam = Family(family) if family else None
    tsk = Task(task) if task else None
    return [
        s for s in MODEL_ZOO.values()
        if (fam is None or s.family == fam) and (tsk is None or s.task == tsk)
    ]


# One schema per medallion layer, named <project>_<numbered_layer> so all the
# project's schemas group together and sort landing -> bronze -> silver -> gold.
# The pipeline side keeps its own copy in transformations/_layers.py (that source
# can't import this module); keep the two maps in sync.
LAYER_ORDER = {"landing": "00_landing", "bronze": "01_bronze",
               "silver": "02_silver", "gold": "03_gold"}


@dataclass(frozen=True)
class Paths:
    """UC namespace + volume paths for one run.

    There is one schema per medallion layer, named ``<project>_<numbered_layer>``
    (e.g. ``seeing_models_03_gold``). Table names are bare — the schema carries
    the project and layer — so a full name is ``<catalog>.<layer_schema>.<table>``.
    Names are plain identifiers (no leading digit), so no backticking is required;
    table() still quotes for safety.
    """

    catalog: str
    project: str = "seeing_models"
    volume: str = "cv_data"

    def schema(self, layer: str) -> str:
        """Schema for a layer, e.g. schema('gold') -> seeing_models_03_gold."""
        return f"{self.project}_{LAYER_ORDER[layer]}"

    def table(self, layer: str, name: str) -> str:
        """Fully-qualified table (quoted for safety); bare name in the layer's schema."""
        return f"`{self.catalog}`.`{self.schema(layer)}`.`{name}`"

    @property
    def volume_root(self) -> str:
        # Raw files and working artifacts live in the landing layer's volume.
        return f"/Volumes/{self.catalog}/{self.schema('landing')}/{self.volume}"

    def model_name(self, name: str) -> str:
        """UC registered-model name (a gold-layer output)."""
        return f"{self.catalog}.{self.schema('gold')}.{name}"

    @property
    def manifest_table(self) -> str:
        """Gold per-image training manifest the trainers read for splits."""
        return self.table("gold", "training_manifest")

    @classmethod
    def from_params(cls, p: dict) -> "Paths":
        return cls(p["catalog"], p.get("project", "seeing_models"),
                   p.get("volume", "cv_data"))


def env(name: str, default: str = "") -> str:
    """Small helper for reading notebook/job params passed as env vars."""
    return os.environ.get(name, default)
