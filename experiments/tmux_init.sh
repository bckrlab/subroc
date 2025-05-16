#!/bin/bash

SESSIONNAME="experiments"
tmux has-session -t $SESSIONNAME &> /dev/null

if [ $? != 0 ] 
 then
    tmux new-session -s $SESSIONNAME -n resource-monitor -d
    tmux send-keys -t $SESSIONNAME "docker stats" Enter
    tmux split-window -t $SESSIONNAME -v
    tmux send-keys -t $SESSIONNAME "htop" Enter
    tmux new-window -t $SESSIONNAME
    tmux send-keys -t $SESSIONNAME "bash experiments/run/start.sh -n exp-0 -m 500g -w 504g -c 20 -s 0 -e 99" Enter
fi

tmux attach -t $SESSIONNAME
