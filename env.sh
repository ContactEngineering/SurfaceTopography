#! /bin/sh

ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="$ROOT:$ROOT/build/lib.$PLATFORM:$PYTHONPATH"
