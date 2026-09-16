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

    YOLO = "yolo"      # Ultralytics YOLO11, one-stage anchor-free
    RESNET = "resnet"  # torchvision R-CNN, ResNet-50 FPN backbone, two-stage
    CNN = "cnn"        # lightweight one-stage CNN baseline


class Task(str, Enum):
    """What the model predicts."""

    DETECT = "detect"  # bounding boxes (golfer, club head, ball)
    POSE = "pose"      # 17 person keypoints -> limb lines (the swing overlay)


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


# Sensible small-but-real defaults; override per-run via notebook parameters.
_DETECT_HP = {"epochs": 50, "imgsz": 640, "batch": 16, "lr": 1e-3}
_POSE_HP = {"epochs": 80, "imgsz": 640, "batch": 16, "lr": 1e-3}

_ZOO = [
    # --- YOLO11 (one-stage) -------------------------------------------------
    ModelSpec("yolo_detect", Family.YOLO, Task.DETECT, "yolo11n.pt",
              "YOLO11-nano detector — golfer / club / ball boxes.", dict(_DETECT_HP)),
    ModelSpec("yolo_pose", Family.YOLO, Task.POSE, "yolo11n-pose.pt",
              "YOLO11-nano pose — 17 keypoints, draws the swing limb lines.", dict(_POSE_HP)),

    # --- ResNet-50 FPN R-CNN (two-stage) ------------------------------------
    ModelSpec("resnet_detect", Family.RESNET, Task.DETECT, "fasterrcnn_resnet50_fpn_v2",
              "Faster R-CNN, ResNet-50 FPN backbone — high-accuracy detector.", dict(_DETECT_HP)),
    ModelSpec("resnet_pose", Family.RESNET, Task.POSE, "keypointrcnn_resnet50_fpn",
              "Keypoint R-CNN, ResNet-50 FPN — 17-keypoint pose for limb lines.", dict(_POSE_HP)),

    # --- Lightweight CNN baseline (one-stage) -------------------------------
    ModelSpec("cnn_detect", Family.CNN, Task.DETECT, "ssdlite320_mobilenet_v3_large",
              "SSDlite MobileNetV3 — fast, small CNN detector baseline.",
              {**_DETECT_HP, "batch": 32}),
    # torchvision has no lightweight keypoint model; YOLO/ResNet carry the pose
    # comparison. Drop in a MobileNet-backed keypoint head here to extend it.
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


@dataclass(frozen=True)
class Paths:
    """UC namespace + volume paths for one run. Built from notebook parameters."""

    catalog: str
    schema: str
    volume: str
    table_prefix: str

    @property
    def volume_root(self) -> str:
        return f"/Volumes/{self.catalog}/{self.schema}/{self.volume}"

    def table(self, layer: str, name: str) -> str:
        """Fully-qualified, backtick-quoted medallion table.

        `layer` is bronze | silver | gold. Names start with a digit (the
        table_prefix), so they must be backtick-quoted in SQL.
        """
        return f"`{self.catalog}`.`{self.schema}`.`{self.table_prefix}_{layer}_{name}`"

    @property
    def manifest_table(self) -> str:
        """Gold per-image training manifest the trainers read for splits."""
        return self.table("gold", "training_manifest")

    def model_name(self, key: str) -> str:
        """UC registered-model name for a model spec key."""
        return f"{self.catalog}.{self.schema}.{self.table_prefix}_{key}"

    @classmethod
    def from_params(cls, p: dict) -> "Paths":
        return cls(p["catalog"], p["schema"], p["volume"], p["table_prefix"])


def env(name: str, default: str = "") -> str:
    """Small helper for reading notebook/job params passed as env vars."""
    return os.environ.get(name, default)
