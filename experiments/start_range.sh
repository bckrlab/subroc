#!/bin/bash

# Ensure two arguments are passed
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <a> <b>"
    exit 1
fi

# Read arguments
a=$1
b=$2

# Check if arguments are integers
if ! [[ "$a" =~ ^-?[0-9]+$ && "$b" =~ ^-?[0-9]+$ ]]; then
    echo "Error: Both arguments must be integers."
    exit 1
fi

# Name of the tmux session
SESSION_NAME="experiments"

# Loop through the range [a, b]
for ((i=a; i<=b; i++)); do
    WINDOW_NAME="exp-$i"
    COMMAND="bash experiments/run/start.sh -n exp-$i -m 200g -w 204g -c 20 -s 0 -e 99"
    
    # Create a new tmux window and run the command
    tmux new-window -t "$SESSION_NAME" -n "$WINDOW_NAME" "$COMMAND"
    echo "started $WINDOW_NAME"

    # Sleep for 60 seconds with a countdown
    for ((t=60; t>0; t--)); do
        printf "\rWaiting for %02d seconds before creating the next window..." "$t"
        sleep 1
    done
    echo "" # New line after countdown
done
