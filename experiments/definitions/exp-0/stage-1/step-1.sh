#!/bin/bash

# importing util functions such as prefixed_echo
. "$EXPERIMENT_BASH_UTIL_PATH"/util.sh

prefixed_echo "Running step-1"

NOTEBOOKS_OUTPUT_PATH="$STAGE_OUTPUT_PATH/notebooks"
mkdir "$NOTEBOOKS_OUTPUT_PATH"

NOTEBOOK_PATH="$EXPERIMENT_CODE_PATH/notebooks/misc/saying_hello.ipynb"

papermill "$NOTEBOOK_PATH" "$NOTEBOOKS_OUTPUT_PATH/step-1_$(basename "$NOTEBOOK_PATH")"
