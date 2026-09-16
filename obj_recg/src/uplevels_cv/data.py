"""Dataset access on the UC Volume + format conversion.

Data is namespaced by sport. Ground truth is stored in COCO format (one JSON per
split) so both torchvision (native COCO) and Ultralytics (YOLO-format export)
can consume it. Volume layout under ``paths.volume_root``:

    raw/<sport>/           # original videos / frame dumps
    images/<sport>/<split>/# extracted frames
    labels/<sport>/        # COCO json: instances_<split>.json, person_keypoints_<split>.json
    games/<sport>/         # game results for team-performance labels (CSV/parquet)
    splits/<sport>/        # yolo-format export written by export_yolo_dataset()
    artifacts/             # per-run model files, plots, downloaded checkpoints
    predictions/           # scored overlays

Frame extraction and labeling are a TODO; this module provides the paths, the
COCO loader, and the YOLO exporter so training is ready once labeled data lands.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from uplevels_cv.config import Paths
from uplevels_cv.skeleton import KEYPOINT_NAMES, SKELETON
from uplevels_cv.sports import SportSpec, sport_spec


@dataclass
class DataConfig:
    paths: Paths
    sport: str = "golf"
    train_split: str = "train"
    val_split: str = "val"

    @property
    def spec(self) -> SportSpec:
        return sport_spec(self.sport)

    @property
    def num_detect_classes(self) -> int:
        return self.spec.num_detect_classes

    def images_dir(self, split: str) -> str:
        return os.path.join(self.paths.volume_root, "images", self.sport, split)

    def coco_json(self, split: str, keypoints: bool) -> str:
        stem = "person_keypoints" if keypoints else "instances"
        return os.path.join(self.paths.volume_root, "labels", self.sport, f"{stem}_{split}.json")

    def manifest(self, spark):
        """The gold training manifest (image_id, file_name, split, sport, counts).

        Built by the medallion pipeline; the split assignment lives here so every
        model trains/evaluates on the same partition. Filter by ``sport`` and
        ``split`` to get each set's frames.
        """
        return spark.table(self.paths.manifest_table).where(f"sport = '{self.sport}'")


def load_coco(path: str) -> dict:
    """Load a COCO-format annotation file from the volume."""
    with open(path) as f:
        return json.load(f)


def ensure_volume_layout(paths: Paths) -> None:
    """Create the volume subdirectories if they don't exist (idempotent)."""
    for sub in ("raw", "images", "labels", "splits", "artifacts", "predictions"):
        os.makedirs(os.path.join(paths.volume_root, sub), exist_ok=True)


def export_yolo_dataset(cfg: DataConfig, keypoints: bool) -> str:
    """Convert COCO annotations to the YOLO/Ultralytics layout and write a
    ``data.yaml``. Returns the path to that yaml (what YOLO.train(data=...) wants).

    Ultralytics needs per-image ``.txt`` label files and a dataset yaml, not
    COCO json. This writes the yaml describing the splits and keypoint shape;
    the per-image label conversion loop is marked TODO — fill it in once the
    COCO files exist so we don't guess at your category ids.
    """
    import yaml

    out = os.path.join(cfg.paths.volume_root, "splits", cfg.sport)
    os.makedirs(out, exist_ok=True)

    data_yaml = {
        "path": out,
        "train": cfg.images_dir(cfg.train_split),
        "val": cfg.images_dir(cfg.val_split),
        "names": dict(enumerate(cfg.spec.detect_classes)),  # sport detection classes
    }
    if keypoints:
        data_yaml["kpt_shape"] = [len(KEYPOINT_NAMES), 3]  # (num_kpts, xyv)
        data_yaml["flip_idx"] = _coco_flip_index()
        data_yaml["skeleton"] = [list(e) for e in SKELETON]

    yaml_path = os.path.join(out, "keypoints.yaml" if keypoints else "detect.yaml")
    with open(yaml_path, "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    # TODO: iterate load_coco(cfg.coco_json(split, keypoints)) and write one
    #       <image>.txt per frame in YOLO format (class cx cy w h [kp_x kp_y v]*).
    return yaml_path


def _coco_flip_index() -> list[int]:
    """Left/right keypoint swap used for horizontal-flip augmentation."""
    idx = []
    for name in KEYPOINT_NAMES:
        if name.startswith("left_"):
            idx.append(KEYPOINT_NAMES.index("right_" + name[5:]))
        elif name.startswith("right_"):
            idx.append(KEYPOINT_NAMES.index("left_" + name[6:]))
        else:
            idx.append(KEYPOINT_NAMES.index(name))
    return idx
