#!/bin/sh

if [ -s /env-mount/env.sh ]; then
  echo "Sourcing env.sh"
  . /env-mount/env.sh
else
  echo "env.sh missing or empty"
fi

if [ -f "$KIT_INPUTS_FILE" ]; then
  echo "Found input file, processing…"
  # do work
else
  echo "No input file supplied, continuing without it"
fi

exec "$@"
