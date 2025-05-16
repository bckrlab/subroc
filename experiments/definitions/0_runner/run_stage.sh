#!/bin/bash

# importing util functions such as prefixed_echo
. "$EXPERIMENT_BASH_UTIL_PATH"/util.sh

# list the bash scripts in the stage dir, which define the stage steps
STAGE_STEPS=$(find "$1" -type f -name "*.sh" -print)

# start stage steps in new tmux sessions each
IFS=$'\n'; set -f  # split the stage step paths by newlines, thereby allowing spaces in filenames
for STAGE_STEP in $STAGE_STEPS; do
  STEP_NAME=$(basename "$STAGE_STEP")
  prefixed_echo "Starting new tmux window \"$STEP_NAME\" to run the experiment step in."
  tmux new-window -d -n "$STEP_NAME" "bash $STAGE_STEP 2>&1 | tee $STAGE_LOG_PATH/$STEP_NAME.log"  # run the stage step in a new tmux window and log the terminal output
done
unset IFS; set +f

prefixed_echo "Done starting all steps."
