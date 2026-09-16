"""Team-performance prediction — tabular models over CV-derived features.

Flow: the medallion builds gold feature tables from the pose/detection outputs
and game results; this module trains one model per target (XGBoost + MLflow
autolog), registers each to Unity Catalog with a @prod alias, and batch-scores
them into a gold predictions table.

Targets are a registry (``TARGETS``), like the CV model zoo. Add a target — a new
use case — by appending one ``PredictionTarget``; training and scoring pick it up
with no other change. Feature and label columns depend on your data and are
marked TODO; the training/registration/scoring flow around them is wired.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from uplevels_cv.config import Paths


class TargetType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class Scope(str, Enum):
    TEAM_GAME = "team_game"  # one row per team per game
    ATHLETE = "athlete"      # one row per athlete per window
    RECRUIT = "recruit"      # one row per recruit


@dataclass(frozen=True)
class PredictionTarget:
    key: str
    target_type: TargetType
    scope: Scope
    label_col: str
    feature_table: str  # gold table suffix, e.g. "team_game_features"
    description: str
    metric: str         # primary eval metric

    @property
    def is_classification(self) -> bool:
        return self.target_type is TargetType.CLASSIFICATION


_TARGETS = [
    # Game outcome + score
    PredictionTarget("win_probability", TargetType.CLASSIFICATION, Scope.TEAM_GAME,
                     "win", "team_game_features", "Probability the team wins.", "roc_auc"),
    PredictionTarget("points_for", TargetType.REGRESSION, Scope.TEAM_GAME,
                     "points_for", "team_game_features", "Team score (points/goals/runs).", "rmse"),
    PredictionTarget("points_against", TargetType.REGRESSION, Scope.TEAM_GAME,
                     "points_against", "team_game_features", "Opponent score allowed.", "rmse"),
    # Efficiency ratings
    PredictionTarget("off_efficiency", TargetType.REGRESSION, Scope.TEAM_GAME,
                     "off_efficiency", "team_game_features",
                     "Offensive efficiency (per possession/drive).", "rmse"),
    PredictionTarget("def_efficiency", TargetType.REGRESSION, Scope.TEAM_GAME,
                     "def_efficiency", "team_game_features",
                     "Defensive efficiency (per possession/drive).", "rmse"),
    # Player performance index
    PredictionTarget("player_index", TargetType.REGRESSION, Scope.ATHLETE,
                     "player_rating", "athlete_metrics",
                     "Per-athlete performance index from biomechanics + events.", "rmse"),
    # Recruiting / scholarship ROI
    PredictionTarget("scholarship_roi", TargetType.REGRESSION, Scope.RECRUIT,
                     "roi", "athlete_metrics",
                     "Projected athlete contribution vs. scholarship cost.", "rmse"),
]

TARGETS: dict[str, PredictionTarget] = {t.key: t for t in _TARGETS}

# Columns that identify rows rather than describe them; excluded from features.
KEY_COLS = {"sport", "team", "opponent", "game_id", "game_date", "athlete_id",
            "recruit_id", "season"}


def targets_for(scope: Scope | str | None = None) -> list[PredictionTarget]:
    """Filter the target registry by scope. No filter returns everything."""
    scp = Scope(scope) if scope else None
    return [t for t in TARGETS.values() if scp is None or t.scope == scp]


def _entity_col(target: PredictionTarget) -> str:
    """Id column score_targets emits as entity_id, by scope (game vs. athlete)."""
    return "game_id" if target.scope is Scope.TEAM_GAME else "athlete_id"


def missing_columns(spark, paths: Paths, target: PredictionTarget, sport: str) -> list[str]:
    """Columns `target` needs but its gold feature table doesn't have yet.

    Checked against the live table so a not-yet-populated target is skipped
    instead of crashing the run: the athlete/recruit targets need player_rating /
    roi / athlete_id, which gold.athlete_metrics doesn't emit until its TODO
    lands. Returns the missing names (empty = ready); a missing or unreadable
    feature table counts as not ready.
    """
    try:
        cols = set(spark.table(paths.table("gold", target.feature_table)).columns)
    except Exception:  # noqa: BLE001 — absent or unreadable table means "not ready"
        return [f"{target.feature_table} (table)"]
    return sorted({target.label_col, _entity_col(target)} - cols)


def _feature_frame(spark, paths: Paths, target: PredictionTarget, sport: str):
    """Load a target's gold feature table for a sport as a pandas frame, split
    into (X, y). Feature columns = every non-key, non-label column.
    """
    pdf = spark.table(paths.table("gold", target.feature_table)).where(
        f"sport = '{sport}'").toPandas()
    feature_cols = [c for c in pdf.columns if c not in KEY_COLS and c != target.label_col]
    # TODO: confirm feature_cols once the gold feature schema is populated; drop
    #       leakage columns and cast categoricals as needed.
    return pdf[feature_cols], pdf[target.label_col], feature_cols


def train_target(spark, paths: Paths, target: PredictionTarget, *, sport: str,
                 experiment_root: str) -> dict:
    """Train one target model and register it to Unity Catalog.

    Steps:
      1. Load features + label for the sport from the gold feature table.
      2. Train/test split.
      3. Train XGBoost (classifier or regressor by target type) with autolog.
      4. Score the primary metric on the test split.
      5. Register the model to UC as <prefix>_<sport>_teamperf_<key> and move @prod.
    """
    import mlflow
    from mlflow import MlflowClient
    from sklearn.model_selection import train_test_split

    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(f"{experiment_root}/team_performance")
    mlflow.xgboost.autolog(log_input_examples=True)

    X, y, feature_cols = _feature_frame(spark, paths, target, sport)          # 1
    stratify = y if target.is_classification else None
    X_tr, X_te, y_tr, y_te = train_test_split(                                # 2
        X, y, test_size=0.2, random_state=42, stratify=stratify)

    model_name = paths.model_name(f"{sport}_teamperf_{target.key}")
    with mlflow.start_run(run_name=f"{sport}_{target.key}"):
        mlflow.log_params({"sport": sport, "target": target.key,
                           "scope": target.scope.value, "n_features": len(feature_cols)})

        if target.is_classification:                                          # 3
            from xgboost import XGBClassifier
            from sklearn.metrics import roc_auc_score

            model = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05)
            model.fit(X_tr, y_tr)
            score = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])      # 4
        else:
            from xgboost import XGBRegressor
            from sklearn.metrics import root_mean_squared_error

            model = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05)
            model.fit(X_tr, y_tr)
            score = root_mean_squared_error(y_te, model.predict(X_te))        # 4

        mlflow.log_metric(target.metric, float(score))
        info = mlflow.xgboost.log_model(model, name="model",                  # 5
                                        registered_model_name=model_name)
        MlflowClient(registry_uri="databricks-uc").set_registered_model_alias(
            model_name, "prod", info.registered_model_version)

    return {"target": target.key, "sport": sport, target.metric: round(float(score), 4),
            "model_version": info.registered_model_version}


def score_targets(spark, paths: Paths, *, sport: str) -> str | None:
    """Batch-score every ready @prod target model over the latest features.

    Writes long-format rows (sport, scope, target, entity_id, prediction) to the
    gold predictions table, replacing only this sport's rows. Long format keeps
    the table open to new targets without schema changes. Targets whose gold
    feature table lacks the columns they need are skipped — train_target skips
    them too, so no @prod model exists to load. Returns the table name, or None
    when no target is ready.
    """
    import mlflow
    from pyspark.sql import functions as F

    frames = []
    for key, target in TARGETS.items():
        if missing_columns(spark, paths, target, sport):
            continue
        model_uri = f"models:/{paths.model_name(f'{sport}_teamperf_{key}')}@prod"
        feats = spark.table(
            paths.table("gold", target.feature_table)).where(f"sport = '{sport}'")
        feature_cols = [c for c in feats.columns if c not in KEY_COLS and c != target.label_col]
        udf = mlflow.pyfunc.spark_udf(spark, model_uri, env_manager="local")
        scored = feats.withColumn("prediction", udf(*[feats[c] for c in feature_cols])).select(
            F.lit(sport).alias("sport"), F.lit(target.scope.value).alias("scope"),
            F.lit(key).alias("target"), F.col(_entity_col(target)).alias("entity_id"),
            F.col("prediction"), F.current_timestamp().alias("scored_at"))
        frames.append(scored)

    if not frames:
        return None

    out = frames[0]
    for f in frames[1:]:
        out = out.unionByName(f)

    table = paths.table("gold", "team_predictions")  # backtick-quoted schema
    (out.write.mode("overwrite").partitionBy("sport")
     .option("replaceWhere", f"sport = '{sport}'").option("mergeSchema", "true")
     .saveAsTable(table))
    return table
