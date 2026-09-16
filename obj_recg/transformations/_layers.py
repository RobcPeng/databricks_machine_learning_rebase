"""Shared layer map + name helper for the medallion transformations.

The pipeline (resources/pipeline.yml) globs transformations/**, and that source
can't import uplevels_cv.config — the src package isn't on the pipeline's path.
So the four transformation files import LAYER_ORDER and fqn from here instead of
each redefining them. Keep this map in sync with the copy in
uplevels_cv/config.py, which serves the notebook/training side.
"""

# One schema per medallion layer, named <project>_<numbered_layer>, so the
# project's schemas group together and sort landing -> bronze -> silver -> gold.
LAYER_ORDER = {"landing": "00_landing", "bronze": "01_bronze",
               "silver": "02_silver", "gold": "03_gold"}


def make_fqn(catalog: str, project: str):
    """Return an ``fqn(layer, table)`` bound to this pipeline's catalog + project.

    Schema names lead with a digit, so the fully-qualified name is backtick-quoted.
    """
    def fqn(layer: str, table: str) -> str:
        return f"`{catalog}`.`{project}_{LAYER_ORDER[layer]}`.`{table}`"

    return fqn
