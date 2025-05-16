#!/bin/bash

# Set default values for options.
EXPERIMENT_NAME="$1"
HOST_CACHE_PATH="$(pwd)/experiments/cache"
HOST_OUTPUT_PATH="$(pwd)/experiments/outputs"
MEMORY_LIMIT_OPTION=""
SWAP_MEMORY_LIMIT_OPTION=""
CPU_LIMIT_OPTION=""
START_STAGE="0"
END_STAGE="999"

# Receive options. (credits to the accepted answer in https://unix.stackexchange.com/questions/129391/passing-named-arguments-to-shell-scripts)
while getopts "n::a::o::m::w::c::s::e::" opt; do
  case $opt in
    n) EXPERIMENT_NAME="$OPTARG"
    ;;
    a) HOST_CACHE_PATH="$OPTARG"
    ;;
    o) HOST_OUTPUT_PATH="$OPTARG"
    ;;
    m) MEMORY_LIMIT_OPTION="--memory=$OPTARG"
    ;;
    w) SWAP_MEMORY_LIMIT_OPTION="--memory-swap=$OPTARG"
    ;;
    c) CPU_LIMIT_OPTION="--cpus=$OPTARG"
    ;;
    s) START_STAGE="$OPTARG"
    ;;
    e) END_STAGE="$OPTARG"
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
echo "Starting experiment container with..."
echo "- experiment name: $EXPERIMENT_NAME"
echo "- cache directory: $HOST_CACHE_PATH"
echo "- output directory: $HOST_OUTPUT_PATH"
echo "- container memory limit option: $MEMORY_LIMIT_OPTION"
echo "- container swap memory limit option: $SWAP_MEMORY_LIMIT_OPTION"
echo "- container cpu limit option: $CPU_LIMIT_OPTION"
echo "- number of stages to skip: $START_STAGE"
echo "- maximal number of stages to run: $END_STAGE"

# Environment variables for inside the docker container.
# Note that the specified paths must be correct inside the container,
# while paths and directory names may differ outside of the container.
EXPERIMENT_OUTPUT_PATH="/experiment/output"
CACHE_PATH="/cache"

# If it exists, remove the container with the same name.
docker container remove $EXPERIMENT_NAME

# Run the container.
docker run \
-v "$HOST_CACHE_PATH/":$CACHE_PATH \
-v "$HOST_OUTPUT_PATH/$EXPERIMENT_NAME":$EXPERIMENT_OUTPUT_PATH \
--env EXPERIMENT_NAME=$EXPERIMENT_NAME \
--env START_STAGE=$START_STAGE \
--env END_STAGE=$END_STAGE \
--name $EXPERIMENT_NAME \
$MEMORY_LIMIT_OPTION \
$SWAP_MEMORY_LIMIT_OPTION \
$CPU_LIMIT_OPTION \
"subroc-experiments"
