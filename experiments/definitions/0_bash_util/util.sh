#!/bin/bash

prefixed_echo () {
	echo -e "[$(basename "$0")] ($(date)) $1"
}

prefixed_echo_line_before() {
	echo -e -n "\e[1A\e[K"
	prefixed_echo "$1"
}
