#!/bin/bash

PARENT_PATH="$(dirname "${BASH_SOURCE[0]}")"

cp "$PARENT_PATH/template.tex" "$STAGE_OUTPUT_PATH"

lualatex -shell-escape -halt-on-error -interaction=nonstopmode -output-directory "$STAGE_OUTPUT_PATH" "$STAGE_OUTPUT_PATH/template.tex"
