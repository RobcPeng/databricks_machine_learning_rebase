"""Gold (03) — training-ready tables for the CV models and predictive models.

Publishes to the gold schema (<project>_03_gold):

  training_manifest  : per-image manifest (sport, split, counts) — CV trainers
      read it for the split assignment.
  athlete_metrics    : per-instance CV features — feeds player_index / scholarship_roi.
  team_game_features : per team-game features + labels — feeds the team-game targets.

The model-comparison and team-prediction leaderboards are also gold tables,
written from MLflow runs by compare.py and team_performance.py.
"""

from pyspark import pipelines as dp
from pyspark.sql import Window
from pyspark.sql import functions as F

from _layers import make_fqn

CATALOG = spark.conf.get("catalog")  # noqa: F821
PROJECT = spark.conf.get("project")  # noqa: F821

fqn = make_fqn(CATALOG, PROJECT)


@dp.materialized_view(
    name=fqn("gold", "training_manifest"),
    comment="Per-image training manifest: sport, split, instance/keypoint counts.",
)
def gold_training_manifest():
    per_image = (
        spark.read.table(fqn("silver", "keypoints"))  # noqa: F821
        .groupBy("sport", "image_id")
        .agg(
            F.count("*").alias("num_instances"),
            F.round(F.avg("num_visible_keypoints"), 2).alias("avg_visible_keypoints"),
            F.round(F.sum("bbox_area"), 1).alias("total_bbox_area"),
        )
    )
    images = spark.read.table(fqn("silver", "images"))  # noqa: F821
    return (
        images.join(per_image, ["sport", "image_id"], "left")
        .na.fill(0, ["num_instances", "avg_visible_keypoints", "total_bbox_area"])
        .withColumn("has_pose", F.col("avg_visible_keypoints") > 0)
    )


@dp.materialized_view(
    name=fqn("gold", "athlete_metrics"),
    comment="Per-instance CV features (coverage, bbox, biomechanics angles).",
)
def gold_athlete_metrics():
    kp = spark.read.table(fqn("silver", "keypoints"))  # noqa: F821
    return kp.select(
        "sport", "image_id", "category_id", "bbox_area", "num_visible_keypoints",
    )
    # TODO: compute the sport's biomechanics angles (sports.SportSpec.angles via
    #       skeleton.joint_angle) as columns from the keypoints array, add an
    #       athlete_id via roster linkage, and a player_rating label. Those three
    #       unlock the player_index and scholarship_roi targets.


@dp.materialized_view(
    name=fqn("gold", "team_game_features"),
    comment="Per team-game features + labels for the team-performance targets.",
)
def gold_team_game_features():
    games = spark.read.table(fqn("silver", "games"))  # noqa: F821

    # Leakage-safe form features: rolling average over the team's PRIOR games
    # only (rowsBetween excludes the current row). These are model inputs; the
    # current-game columns (points_for/against, win, efficiency) are the labels.
    w = Window.partitionBy("sport", "team").orderBy("game_date").rowsBetween(-5, -1)
    return (
        games
        .withColumn("form_points_for", F.avg("points_for").over(w))
        .withColumn("form_points_against", F.avg("points_against").over(w))
        .withColumn("form_win_rate", F.avg("win").over(w))
        # TODO: join aggregated athlete_metrics per (sport, team, game) to add
        #       CV-derived team-strength features.
    )
