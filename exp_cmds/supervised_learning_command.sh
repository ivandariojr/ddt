#!/usr/bin/env bash

cd ..

export PYTHONPATH=./ntpg

script_execute() {
python sl_pipeline.py --dataset $1 --model $2
}

export -f script_execute
#
parallel -j10 script_execute \
::: ttt cesarean cancer \
::: ddt tree
