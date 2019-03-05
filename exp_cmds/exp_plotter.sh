#!/usr/bin/env bash

cd ..

export PYTHONPATH=./ntpg

script_execute() {
python plot.py --exp_root $1 --plt_tag $2
}

export -f script_execute
#
parallel -j10 script_execute \
::: tensorboards/final_exp/LunarLander \
tensorboards/final_exp/CartPole \
tensorboards/final_exp/Acrobot \
::: Mean_History_Reward Inst_Mean_Eval_Reward Reward