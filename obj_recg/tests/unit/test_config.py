"""Config/zoo tests — no torch required, so they run in plain CI."""

from uplevels_cv.config import Family, MODEL_ZOO, Paths, Task, specs_for


def test_zoo_covers_all_families_and_both_tasks():
    families = {s.family for s in MODEL_ZOO.values()}
    assert families == {Family.YOLO, Family.RESNET, Family.CNN}
    tasks = {s.task for s in MODEL_ZOO.values()}
    assert tasks == {Task.DETECT, Task.POSE}


def test_keys_are_unique_and_match_dict():
    assert all(key == spec.key for key, spec in MODEL_ZOO.items())


def test_specs_for_filters():
    assert {s.key for s in specs_for(family="yolo")} == {"yolo_detect", "yolo_pose"}
    assert all(s.task is Task.POSE for s in specs_for(task=Task.POSE))
    assert len(specs_for()) == len(MODEL_ZOO)


def test_paths_quote_digit_leading_medallion_tables():
    p = Paths("rpeng_upleveling", "object_and_vision", "cv_data", "00_seeing_models")
    assert p.table("gold", "model_comparison") == (
        "`rpeng_upleveling`.`object_and_vision`.`00_seeing_models_gold_model_comparison`")
    assert p.table("bronze", "raw_labels") == (
        "`rpeng_upleveling`.`object_and_vision`.`00_seeing_models_bronze_raw_labels`")
    assert p.manifest_table.endswith("00_seeing_models_gold_training_manifest`")
    assert p.volume_root == "/Volumes/rpeng_upleveling/object_and_vision/cv_data"
    assert p.model_name("yolo_pose") == (
        "rpeng_upleveling.object_and_vision.00_seeing_models_yolo_pose")
