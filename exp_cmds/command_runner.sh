#!/usr/bin/env bash

cd ..

export PYTHONPATH=./ntpg

script_execute() {
python pg_pipeline.py --env_name $1 --alg 'ppo' --use-gae \
--model_type ddt --ddt_depth $2 --lr 3e-2 --reg_init $3 --reg_mult $4 \
--sig_alpha_init $5 --sig_alpha_mult $6 --sig_alpha_inc $7\
 --save_model_interval 50 --plot_tree --cuda --seed $8  \
 --param_init_func uniform --reg_type leaf --reg_mode $9
}

export -f script_execute
#
parallel -j10 script_execute \
::: Acrobot-v1 LunarLander-v2 CartPole-v0 \
::: 4 6 \
::: 0.0001 \
::: 1.01859 \
::: 1 \
::: 1.0 \
::: 0 \
::: 1 2 3 4 5 \
::: never converge