#!/bin/sh

if [ -s /env-mount/env.sh ]; then
  echo "Sourcing env.sh"
  . /env-mount/env.sh
else
  echo "env.sh missing or empty"
fi

exec "$@"
