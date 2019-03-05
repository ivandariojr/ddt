#!/usr/bin/env bash

cd ..

export PYTHONPATH=./ntpg

script_execute() {
python pg_pipeline.py --env_name $1 --alg 'ppo' --use-gae \
--model_type mlp --lr 3e-4 --exp_root tensorboards/mlp_baselines \
--save_model_interval 100 --cuda --seed $2 --param_init_func random
}

export -f script_execute

parallel -j12 script_execute \
::: CartPole-v0 Acrobot-v1 LunarLander-v2 \
::: 1 2 3 4 5 