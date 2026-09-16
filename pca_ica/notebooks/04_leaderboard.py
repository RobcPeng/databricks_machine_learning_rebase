# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Leaderboard (gold) + comparison views
# MAGIC
# MAGIC Read `*_leaderboard_gold`, write a tidy `*_model_comparison_gold` summary
# MAGIC (best test ROC-AUC per dataset × model × reducer) for dashboards/Genie, and
# MAGIC render model × reducer heatmaps.

# COMMAND ----------

import os
import sys

_ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
_nb_path = _ctx.notebookPath().get()
FILES_ROOT = os.path.dirname(os.path.dirname("/Workspace" + _nb_path))
sys.path.insert(0, os.path.join(FILES_ROOT, "src"))
CONF_PATH = os.path.join(FILES_ROOT, "conf", "experiments.yml")

dbutils.widgets.text("catalog", "rpeng_upleveling")
dbutils.widgets.text("schema", "dimensionality_reduction")
dbutils.widgets.text("table_prefix", "00_experiment_type")

from dr_bench import load_config

cfg = load_config(
    CONF_PATH,
    catalog=dbutils.widgets.get("catalog"),
    schema=dbutils.widgets.get("schema"),
    table_prefix=dbutils.widgets.get("table_prefix"),
)

# COMMAND ----------

lb = spark.sql(f"SELECT * FROM {cfg.table('leaderboard_gold')}").toPandas()
display(lb.sort_values(["dataset", "test_roc_auc"], ascending=[True, False]))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Gold: best score per dataset × model × reducer

# COMMAND ----------

comparison = (
    lb.groupby(["dataset", "model", "reducer"], as_index=False)["test_roc_auc"]
    .max()
    .sort_values(["dataset", "test_roc_auc"], ascending=[True, False])
)
spark.createDataFrame(comparison).createOrReplaceTempView("comp_tmp")
spark.sql(
    f"CREATE OR REPLACE TABLE {cfg.table('model_comparison_gold')} AS SELECT * FROM comp_tmp"
)
print(f"Wrote {cfg.table('model_comparison_gold')}")
display(comparison)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Heatmap: does reducing dimensions help each model?

# COMMAND ----------

import matplotlib.pyplot as plt

for ds in cfg.datasets:
    sub = lb[lb.dataset == ds]
    if sub.empty:
        continue
    pivot = sub.pivot_table(
        index="model", columns="reducer", values="test_roc_auc", aggfunc="max"
    )
    print(f"\n=== {ds}: max test ROC-AUC by model × reducer ===")
    print(pivot.round(3).to_string())

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    for r in range(pivot.shape[0]):
        for c in range(pivot.shape[1]):
            v = pivot.values[r, c]
            if v == v:  # skip NaN
                ax.text(c, r, f"{v:.2f}", ha="center", va="center", color="white", fontsize=8)
    ax.set_title(f"{ds} — test ROC-AUC")
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.show()
