#!/bin/bash

# importing util functions such as prefixed_echo
. "$EXPERIMENT_BASH_UTIL_PATH"/util.sh

NOTEBOOKS_OUTPUT_PATH="$STAGE_OUTPUT_PATH/notebooks"
mkdir "$NOTEBOOKS_OUTPUT_PATH"

NOTEBOOK_PATH="$EXPERIMENT_CODE_PATH/notebooks/misc/subgroup_result_set_ious.ipynb"
PARENT_PATH="$(dirname "${BASH_SOURCE[0]}")"
STAGE_NAME="$(basename $PARENT_PATH)"

# list the bash scripts in the stage dir, which define the stage steps
PARAMETERIZATIONS=$(find "$PARENT_PATH" -type f -name "*.yaml" -print)

# start parameterizations of stage step in new tmux sessions each
IFS=$'\n'; set -f  # split the parameterization paths by newlines, thereby allowing spaces in filenames
for PARAMETERIZATION in $PARAMETERIZATIONS; do
  PARAMETERIZATION_NAME=$(basename "$PARAMETERIZATION" .yaml)

  prefixed_echo "parameter file content:"
  cat "$PARAMETERIZATION"

  prefixed_echo "Starting new tmux window \"$PARAMETERIZATION_NAME\" to run the parameterized step in."
  tmux new-window -d -n "$PARAMETERIZATION_NAME" "papermill "$NOTEBOOK_PATH" "$NOTEBOOKS_OUTPUT_PATH/"$PARAMETERIZATION_NAME"_$(basename "$NOTEBOOK_PATH")" -f $PARAMETERIZATION 2>&1 | tee $STAGE_LOG_PATH/params_$PARAMETERIZATION_NAME.log"  # run the parameterized stage step in a new tmux window and log the terminal output
  sleep 1
done
unset IFS; set +f

prefixed_echo "Done starting all parameterizations."
