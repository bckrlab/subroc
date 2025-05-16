#!/bin/bash

PARENT_PATH="$(dirname "${BASH_SOURCE[0]}")"

cp "$EXPERIMENT_OUTPUT_PATH/stage-11/with_oe_runtimes_table.tex" "$STAGE_OUTPUT_PATH"
cp "$EXPERIMENT_OUTPUT_PATH/stage-11/no_oe_runtimes_table.tex" "$STAGE_OUTPUT_PATH"
cp "$PARENT_PATH/template.tex" "$STAGE_OUTPUT_PATH"

tlmgr install preview booktabs
pdflatex -halt-on-error -interaction=nonstopmode -output-directory "$STAGE_OUTPUT_PATH" "$STAGE_OUTPUT_PATH/template.tex"
