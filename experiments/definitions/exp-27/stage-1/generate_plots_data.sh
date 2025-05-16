#!/bin/bash

NOTEBOOKS_OUTPUT_PATH="$STAGE_OUTPUT_PATH/notebooks"
mkdir "$NOTEBOOKS_OUTPUT_PATH"

NOTEBOOK_PATH="$EXPERIMENT_CODE_PATH/notebooks/03_experiments/exp-27_generate_plots_data.ipynb"
PARENT_PATH="$(dirname "${BASH_SOURCE[0]}")"
PARAMS_PATH="$PARENT_PATH/params.yaml"

papermill "$NOTEBOOK_PATH" "$NOTEBOOKS_OUTPUT_PATH/$(basename "$NOTEBOOK_PATH")" -f "$PARAMS_PATH"
