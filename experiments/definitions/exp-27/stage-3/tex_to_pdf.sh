#!/bin/bash

PARENT_PATH="$(dirname "${BASH_SOURCE[0]}")"

find $EXPERIMENT_OUTPUT_PATH/stage-1 -name \*.csv -exec cp {} "$STAGE_OUTPUT_PATH" \;
find $EXPERIMENT_OUTPUT_PATH/stage-2 -name \*.csv -exec cp {} "$STAGE_OUTPUT_PATH" \;
cp "$PARENT_PATH/template.tex" "$STAGE_OUTPUT_PATH"

cd "$STAGE_OUTPUT_PATH"
lualatex -shell-escape -halt-on-error -interaction=nonstopmode "$STAGE_OUTPUT_PATH/template.tex"
