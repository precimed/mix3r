#!/bin/bash
set -e

case "$1" in
  "mix3r_int_weights")
    shift
    /opt/conda/bin/python /scripts/mix3r_int_weights.py "$@"
    ;;
  "extract_p")
    shift
    /opt/conda/bin/python /scripts/extract_p.py "$@"
    ;;
  "make_template")
    shift
    /opt/conda/bin/python /scripts/make_template.py "$@"
    ;;
  "make_euler")
    shift
    /opt/conda/bin/Rscript /scripts/make_euler.r "$@"
    ;;
  "python")
    shift
    /opt/conda/bin/python "$@"
    ;;
  "ipython")
    shift
    /opt/conda/bin/ipython "$@"
    ;;
  "jupyter")
    shift
    /opt/conda/bin/jupyter "$@"
    ;;
  "bash")
    shift
    /usr/bin/bash "$@"
    ;;
  *)
    echo "Invalid command. Available commands are: mix3r_int_weights, extract_p, make_template, make_euler, python, ipython, jypyter, bash"
    exit 1
    ;;
esac
