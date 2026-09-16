"""Config/zoo tests — no torch required, so they run in plain CI."""

from uplevels_cv.config import Family, MODEL_ZOO, Paths, Task, specs_for


def test_zoo_covers_all_families_and_tasks():
    families = {s.family for s in MODEL_ZOO.values()}
    assert families == {Family.YOLO, Family.RESNET, Family.CNN,
                        Family.DETR, Family.MMPOSE, Family.MEDIAPIPE}
    tasks = {s.task for s in MODEL_ZOO.values()}
    assert tasks == {Task.DETECT, Task.POSE, Task.POSE3D}


def test_keys_are_unique_and_match_dict():
    assert all(key == spec.key for key, spec in MODEL_ZOO.items())


def test_specs_for_filters():
    assert {s.key for s in specs_for(family="yolo")} == {"yolo_detect", "yolo_pose"}
    assert {s.key for s in specs_for(family=Family.DETR)} == {"rtdetr_detect"}
    assert all(s.task is Task.POSE for s in specs_for(task=Task.POSE))
    assert len(specs_for()) == len(MODEL_ZOO)


def test_only_mediapipe_is_inference_only():
    not_trainable = {s.key for s in MODEL_ZOO.values() if not s.trainable}
    assert not_trainable == {"blazepose_pose3d"}


def test_schema_per_layer_and_bare_table_names():
    p = Paths("rpeng_upleveling", "seeing_models", "cv_data")
    # One schema per layer, <project>_<numbered_layer>; table names are bare.
    assert p.schema("landing") == "seeing_models_00_landing"
    assert p.schema("gold") == "seeing_models_03_gold"
    assert p.table("landing", "raw_labels") == (
        "`rpeng_upleveling`.`seeing_models_00_landing`.`raw_labels`")
    assert p.table("bronze", "keypoints") == (
        "`rpeng_upleveling`.`seeing_models_01_bronze`.`keypoints`")
    assert p.table("silver", "keypoints") == (
        "`rpeng_upleveling`.`seeing_models_02_silver`.`keypoints`")
    assert p.table("gold", "model_comparison") == (
        "`rpeng_upleveling`.`seeing_models_03_gold`.`model_comparison`")
    assert p.manifest_table.endswith("`seeing_models_03_gold`.`training_manifest`")
    assert p.volume_root == "/Volumes/rpeng_upleveling/seeing_models_00_landing/cv_data"
    assert p.model_name("golf_yolo_pose") == (
        "rpeng_upleveling.seeing_models_03_gold.golf_yolo_pose")
