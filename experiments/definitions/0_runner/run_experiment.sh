#!/bin/bash

# importing util functions such as prefixed_echo
. "$EXPERIMENT_BASH_UTIL_PATH"/util.sh

prefixed_echo "Running experiment $EXPERIMENT_NAME"

STAGE_DIRS=$(find "$EXPERIMENT_DEFINITIONS_PATH/$EXPERIMENT_NAME" -type d -name "stage-*" -print0 | sort -z | sed 's/\x0/\n/g')

prefixed_echo "Found stages:"
echo "$STAGE_DIRS"

# run the stages sequentially
STAGE_INDEX=0
IFS=$'\n'; set -f  # split the stage paths by newlines, thereby allowing spaces in filenames
for STAGE_DIR in $STAGE_DIRS; do
  if [[ $((STAGE_INDEX-START_STAGE)) -ge $END_STAGE ]]; then
    break
  fi

  if [[ $STAGE_INDEX -lt $START_STAGE ]]; then
    ((STAGE_INDEX++))
    continue
  fi

  STAGE_NAME=$(basename "$STAGE_DIR")
  prefixed_echo "Running $STAGE_NAME (index $STAGE_INDEX)"

  STAGE_OUTPUT_PATH="$EXPERIMENT_OUTPUT_PATH/$STAGE_NAME"
  rm -r "$STAGE_OUTPUT_PATH"
  mkdir "$STAGE_OUTPUT_PATH"
  export STAGE_OUTPUT_PATH

  STAGE_LOG_PATH="$STAGE_OUTPUT_PATH/logs"
  mkdir "$STAGE_LOG_PATH"
  export STAGE_LOG_PATH

  tmux new-session -d -s "$STAGE_NAME" "bash $EXPERIMENT_RUNNER_PATH/run_stage.sh $STAGE_DIR 2>&1 | tee $STAGE_LOG_PATH/run_stage.sh.log"  # run the stage in a new tmux session and log the terminal output

  prefixed_echo "Waiting for session to terminate..."
  while tmux list-sessions | grep -q "$STAGE_NAME";
  do
    prefixed_echo_line_before "Waiting for session to terminate..."
    sleep 1
  done

  ((STAGE_INDEX++))
done
unset IFS; set +f

prefixed_echo "Done."
