#!/bin/bash

# Set default values for options.
HOST_DEFINITIONS_PATH="$(pwd)/experiments/definitions"
HOST_DATA_PATH="$(pwd)/experiments/data"
HOST_CODE_PATH="$(pwd)/code"

# Receive options. (credits to the accepted answer in https://unix.stackexchange.com/questions/129391/passing-named-arguments-to-shell-scripts)
while getopts "d::b::p::" opt; do
  case $opt in
    d) HOST_DEFINITIONS_PATH="$OPTARG"
    ;;
    b) HOST_DATA_PATH="$OPTARG"
    ;;
    p) HOST_CODE_PATH="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    exit 1
    ;;
  esac

  case $OPTARG in
    -*) echo "Option $opt needs a valid argument"
    exit 1
    ;;
  esac
done
shift $((OPTIND-1))  # shift positional arguments such that arguments after the options start with 1 again

# Report the accepted configuration.
echo "Building experiment container with..."
echo "- experiment definitions directory: $HOST_DEFINITIONS_PATH"
echo "- data directory: $HOST_DATA_PATH"
echo "- code directory: $HOST_CODE_PATH"

# Environment variables for inside the docker container.
# Note that the specified paths must be correct inside the container,
# while paths and directory names may differ outside of the container.
EXPERIMENT_PATH="/experiment"
EXPERIMENT_CODE_PATH="$EXPERIMENT_PATH/code"
EXPERIMENT_DEFINITIONS_PATH="$EXPERIMENT_PATH/definitions"
EXPERIMENT_DATA_PATH="$EXPERIMENT_PATH/data"
EXPERIMENT_OUTPUT_PATH="$EXPERIMENT_PATH/output"
EXPERIMENT_RUNNER_PATH="$EXPERIMENT_PATH/runner"
EXPERIMENT_BASH_UTIL_PATH="$EXPERIMENT_PATH/bash_util"
CACHE_PATH="/cache"

# make the experiment definition available to the docker builder
BUILD_DEFINITIONS_PATH="$(pwd)/experiments/run/definitions"
mkdir "$BUILD_DEFINITIONS_PATH"
cp -r "$HOST_DEFINITIONS_PATH/." "$BUILD_DEFINITIONS_PATH"

# make the data available to the docker builder
BUILD_DATA_PATH="$(pwd)/experiments/run/data"
mkdir "$BUILD_DATA_PATH"
cp -r "$HOST_DATA_PATH/." "$BUILD_DATA_PATH"

# make the code available to the docker builder
BUILD_CODE_PATH="$(pwd)/experiments/run/code"
mkdir "$BUILD_CODE_PATH"
cp -r "$HOST_CODE_PATH/." "$BUILD_CODE_PATH"

# Build the image.
docker buildx build \
-t "subroc-experiments" \
--build-arg EXPERIMENT_PATH="$EXPERIMENT_PATH" \
--build-arg EXPERIMENT_CODE_PATH="$EXPERIMENT_CODE_PATH" \
--build-arg EXPERIMENT_DEFINITIONS_PATH="$EXPERIMENT_DEFINITIONS_PATH" \
--build-arg EXPERIMENT_DATA_PATH="$EXPERIMENT_DATA_PATH" \
--build-arg EXPERIMENT_OUTPUT_PATH="$EXPERIMENT_OUTPUT_PATH" \
--build-arg EXPERIMENT_RUNNER_PATH="$EXPERIMENT_RUNNER_PATH" \
--build-arg EXPERIMENT_BASH_UTIL_PATH="$EXPERIMENT_BASH_UTIL_PATH" \
--build-arg CACHE_PATH="$CACHE_PATH" \
./experiments/run

# clean up build resources
rm -r "$BUILD_DEFINITIONS_PATH"
rm -r "$BUILD_DATA_PATH"
rm -r "$BUILD_CODE_PATH"
