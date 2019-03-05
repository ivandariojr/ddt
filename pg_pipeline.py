import gym
import math
import random
import numpy as np
import os
import sys
from collections import namedtuple
from itertools import count
import torch
import torch.optim as optim
import pandas as pd
from torch.distributions import Categorical
from models import *
from tensorboardX import SummaryWriter
from PIL import Image
import argparse
from ntpg.algo import A2C_ACKTR, PPO, A2C_ACKTR
from ntpg.envs import make_vec_envs
from ntpg.model import Policy
from ntpg.storage import RolloutStorage
from ntpg.utils import get_vec_normalize
from collections import deque

def progress_bar(current_value, total):
    increments = 50
    percentual = ((current_value/ total) * 100)
    i = int(percentual // (100 / increments ))
    text = "\r[{0: <{1}}] {2:.2f}%".format('=' * i, increments, percentual)
    print(text, end="\n" if percentual == 100 else "")

class Pipeline:

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        #system init
        if torch.cuda.is_available() and self.cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        torch.manual_seed(self.seed)

        if self.record_video:
            os.makedirs(os.path.join(self.video_root, self.video_out), exist_ok=True)
            print('[INFO] Recording Video')

        self.log_root = os.path.join(self.run_log_dir_root, self._create_run_name())
        self.save_model_path = os.path.join(self.log_root, 'models')
        self.save_plots_path = os.path.join(self.log_root, 'plots')
        self.save_env_path = os.path.join(self.log_root, 'envs')
        os.makedirs(self.save_model_path, exist_ok=True)
        if self.plot_tree:
            os.makedirs(self.save_plots_path, exist_ok=True)

        self.model_converged = False
        self._create_env()
        self._create_policy()
        self._create_agent()
        self.rollouts = RolloutStorage(self.n_steps, self.num_processes,
                                  self.envs.observation_space.shape,
                                  self.envs.action_space,
                                  self.actor_critic.recurrent_hidden_state_size)
        #training params
        self.reg = self.reg_init
        self.writer = SummaryWriter(os.path.join(
            self.exp_root, self._create_run_name(), self.run_suffix))
        #storage information
        self.reward_history = deque(maxlen=100)
        self.mean_eval_reward_history = deque(maxlen=10)
        self.mean_history = deque(maxlen=10)
        self.std_history = deque(maxlen=10)

    def _regularize_lambda(self):
        if self.reg_mode == 'never':
            return 0.0
        if self.reg_mode == 'always' or \
            (self.reg_mode == 'converge' and self.model_converged):
            if self.reg_type == 'sum':
                return self.policy.regularization() * self.reg
            elif self.reg_type == 'reg':
                return self.policy.regregularization() * self.reg
            elif self.reg_type == 'leaf':
                return (self.policy.regregularization() + \
                       self.policy.leaf_regularization()) * self.reg
        return 0.0

    def _evaluate_policy(self):
        env_log_dir = '/tmp/gym_eval/'
        try:
            os.makedirs(env_log_dir)
        except OSError:
            import glob
            files = glob.glob(os.path.join(env_log_dir, '*.monitor.csv'))
            for f in files:
                os.remove(f)
        if self.model_type == 'ddt':
            self.policy.make_hard()
        eval_envs = make_vec_envs(self.env_name,
                                  self.seed + self.num_processes,
                                  self.num_processes,
                                  self.gamma,
                                  env_log_dir,
                                  False,
                                  self.device,
                                  False)
        vec_norm = get_vec_normalize(eval_envs)
        if vec_norm is not None:
            vec_norm.eval()
            vec_norm.obs_rms = get_vec_normalize(self.envs).ob_rms
        eval_episode_rewards = []
        obs = eval_envs.reset()
        eval_recurrent_hidden_states = torch.zeros(self.num_processes,
                                                   self.actor_critic.recurrent_hidden_state_size,
                                                   device=self.device)
        eval_masks = torch.zeros(self.num_processes, 1, device=self.device)
        while len(eval_episode_rewards) < 10:
            with torch.no_grad():
                _, action, _, eval_recurrent_hidden_states =\
                self.actor_critic.act(obs, eval_recurrent_hidden_states,
                                      eval_masks, deterministic=True)
                obs, reward, done, infos, = eval_envs.step(action)
                eval_masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                                for done_ in done])
                for info in infos:
                    if 'episode' in info.keys():
                        eval_episode_rewards.append(info['episode']['r'])
        eval_envs.close()
        if self.model_type == 'ddt':
            self.policy.make_soft()
        return eval_episode_rewards


    def _create_env(self):
        # defines true_env, env, n_action, n_state
        if self.env_name == "CartPole-v0":
            self.n_action = 2
            self.n_state = 4
            self.state_labels = ["x", "Dx", "th", "Dth"]
            self.action_labels = ['left', 'right']
            self.cont_act = False
        elif self.env_name == "Acrobot-v1":
            self.n_action = 3
            self.n_state = 6
            self.cont_act = False
            self.state_labels = ["sth1", "cth1", "sth2", "cth2", "Dth1", "Dth2"]
            self.action_labels = ['+1', '0', '-1']
        elif self.env_name == "MountainCar-v0":
            self.n_action = 3
            self.n_state = 2
            self.state_labels = ["x", "Dx"]
            self.cont_act = False
        elif self.env_name == "MountainCarContinuous-v0":
            self.n_action = 1
            self.n_state = 2
            self.state_labels = ["x", "Dx"]
            self.cont_act = True
            self.action_labels = ['push_direction']
        elif self.env_name == "LunarLander-v2":
            # Nop, fire left engine, main engine, right engine
            self.n_action = 4
            self.n_state = 8
            self.state_labels = ["x", "y", "Dx", "Dy", "th", "Dth", "gc1", "gc2"]
            self.action_labels = ['noop', 'left', 'main', 'right', ]
            self.cont_act = False
        elif self.env_name == "LunarLanderContinuous-v2":
            self.n_action = 2
            self.n_state = 8
            self.state_labels = ["x", "y", "Dx", "Dy", "th", "Dth", "gc1", "gc2"]
            self.action_labels = ['main', 'side']
            self.cont_act = True
        env_log_dir = '/tmp/gym/'
        try:
            os.makedirs(env_log_dir)
        except OSError:
            import glob
            files = glob.glob(os.path.join(env_log_dir, '*.monitor.csv'))
            for f in files:
                os.remove(f)
        self.envs = make_vec_envs(self.env_name,
                                  self.seed,
                                  self.num_processes,
                                  self.gamma,
                                  env_log_dir,
                                  False,
                                  self.device,
                                  False)
        if self.cont_act:
            raise Exception("[ERROR] Continuous Action Spaces Not Supported")

    def _create_policy(self):
        if self.model_type == 'ddt':
            self.policy = FDDTN(self.ddt_depth, self.n_state,
                               self.n_action, self.cont_act, self.state_labels,
                                param_init_func[self.param_init_func],
                                self.sig_alpha_init, action_labels=self.action_labels)
        elif self.model_type == 'lin':
            self.policy = LinearPolicy(self.n_state, self.n_action, self.cont_act)
        elif self.model_type == 'mlp':
            self.policy = MLP(self.n_state, self.n_action, self.cont_act)
        else:
            assert False
        self.critic = MLP(self.n_state, 1, True)
        self.critic.to(self.device)
        self.policy.to(self.device)
        self.actor_critic = Policy(self.envs.observation_space.shape,
                                   self.envs.action_space,
                                   lambda: ActorCritic(self.policy, self.critic).to(self.device))
        self.actor_critic.to(self.device)

    def _create_agent(self):
        if self.alg == 'a2c':
            self.agent = A2C_ACKTR(self.actor_critic,
                                   self.value_loss_coef,
                                   self.entropy_coef,
                                   lr=self.lr,
                                   eps=self.eps,
                                   alpha=self.alpha,
                                   max_grad_norm=self.max_grad_norm)
        elif self.alg == 'ppo':
            self.agent = PPO(self.actor_critic,  self.clip_param,
                             self.ppo_epoch, self.num_mini_batch,
                             self.value_loss_coef, self.entropy_coef,
                             lr=self.lr, eps=self.eps,
                             max_grad_norm=self.max_grad_norm,
                             regularize_lambda=lambda: self._regularize_lambda())
        elif self.alg == 'acktr':
            self.agent = A2C_ACKTR(self.actor_critic, self.value_loss_coef,
                                   self.entropy_coef, acktr=True)

    def _create_run_name(self):
        name_list = [ 'env.{}'.format(self.env_name) ]
        if self.model_type == "ddt":
            name_list += [
                'm.{}.{}'.format(self.model_type, self.ddt_depth),
                'reg.b{}.m{}.t{}.md{}'.format(self.reg_init, self.reg_mult,
                                              self.reg_type, self.reg_mode),
                'ceps.{}'.format(self.converge_eps),
                'alpha.b{}.m{}.inc{}'.format(self.sig_alpha_init,
                                             self.sig_alpha_mult,
                                             self.sig_alpha_inc)]
        else:
            name_list += ['m.{}'.format(self.model_type)]
        name_list += [
                'lr.{}'.format(self.lr),
                'alg.{}'.format(self.alg),
                'init.{}'.format(self.param_init_func),
                's.{}'.format(self.seed)
            ]
        run_name = '-'.join(name_list)
        return run_name

    def select_action(self, state):
        p_a = self.policy(state)
        if self.cont_act:
            return p_a, p_a
        dist = Categorical(probs=p_a)
        a = dist.sample()
        return a, p_a

    def save_models(self):
        torch.save(self.policy, os.path.join(self.save_model_path,
                                            'policy_net{}.pth'.format(
                                                self.steps_done)))
    def run(self):
        # torch.autograd.set_detect_anomaly(True)
        obs = self.envs.reset()
        self.rollouts.obs[0].copy_(obs)
        self.rollouts.to(self.device)
        # self.policy_net.tree_to_png('plots/init_pi.svg')

        for i_update in range(self.n_updates):
            for i_step in range(self.n_steps):
                # Sample actions
                with torch.no_grad():
                    value, action, action_log_prob, recurrent_hidden_states = \
                        self.actor_critic.act(
                            self.rollouts.obs[i_step],
                            self.rollouts.recurrent_hidden_states[i_step],
                            self.rollouts.masks[i_step])
                # Obser reward and next obs
                obs, reward, done, infos = self.envs.step(action)
                for info in infos:
                    if 'episode' in info.keys():
                        self.reward_history.append(info['episode']['r'])
                        steps = ((i_update + 1)*self.n_steps + i_step)*self.num_processes
                        self.writer.add_scalar('Reward', info['episode']['r'],
                                               steps)
                # If done then clean the history of observations.
                masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                           for done_ in done])
                self.rollouts.insert(obs, recurrent_hidden_states, action,
                                     action_log_prob, value, reward, masks)
            with torch.no_grad():
                next_value = self.actor_critic.get_value(
                    self.rollouts.obs[-1],
                    self.rollouts.recurrent_hidden_states[-1],
                    self.rollouts.masks[-1]).detach()

            self.rollouts.compute_returns(
                next_value, args.use_gae, args.gamma, args.tau)

            value_loss, action_loss, dist_entropy = self.agent.update(self.rollouts)

            if self.reg_mode == 'always' or \
                (self.reg_mode == 'converge' and self.model_converged):
                self.reg *= self.reg_mult
                self.policy.alpha *= self.sig_alpha_mult
                self.policy.alpha += self.sig_alpha_inc

            self.rollouts.after_update()
            total_steps = (i_update + 1) * self.num_processes * (self.n_steps+1)
            if i_update % self.save_model_interval == 0:
                #TODO: Save models at a regular interval
                if self.plot_tree and self.model_type == 'ddt':
                    self.policy.tree_to_png(
                        os.path.join(self.save_plots_path,
                                     "{}.svg".format(total_steps)))
                    self.policy.hard_tree_to_png(
                        os.path.join(self.save_plots_path,
                                     "{}_hard.svg".format(total_steps)))
            if len(self.reward_history) >= 1 and \
                i_update % self.eval_interval == 0 :
                eval_rewards = self._evaluate_policy()
                e_rw_mean = np.mean(eval_rewards)
                self.writer.add_scalar('Inst Mean Eval Reward', e_rw_mean, total_steps)
                self.mean_eval_reward_history.append(e_rw_mean)

            reward_mean = np.mean(self.reward_history)
            reward_std = np.std(self.reward_history)
            #check for convergence
            if len(self.mean_history) == self.mean_history.maxlen:
                mean_check = np.abs(np.mean(self.mean_history) - reward_mean)\
                             <= self.converge_eps
                std_check = np.abs(np.mean(self.std_history) - reward_std) <=\
                            self.converge_eps
                if mean_check and std_check:
                    self.model_converged = True
            if not self.model_converged:
                self.mean_history.append(reward_mean)
                self.std_history.append(reward_std)
            #log progress
            self.writer.add_scalar('Alpha', self.policy.alpha, total_steps)
            self.writer.add_scalar('Mean History Reward',
                                   reward_mean, total_steps)
            self.writer.add_scalar('Regterm',
                                   self.reg, total_steps)
            self.writer.add_scalar('Converged',
                                   0.0 if not self.model_converged else 1.0,
                                   total_steps)
            self.writer.add_scalar('Std History Reward',
                                   reward_std, total_steps)
            self.writer.add_scalar('Dist Entropy',
                                   dist_entropy, total_steps)
            progress_bar(i_update, self.n_updates)
        if self.plot_tree and self.model_type == 'ddt':
            self.policy.tree_to_png(
                os.path.join(self.save_plots_path,
                             "{}.svg".format(
                                 (self.n_updates+1) * self.num_processes *
                                 (self.n_steps+1))))
            self.policy.hard_tree_to_png(
                os.path.join(self.save_plots_path,
                             "{}_hard.svg".format(
                                 (self.n_updates + 1) * self.num_processes *
                                 (self.n_steps + 1))))
        self.writer.close()


models = ['ddt', 'lin', 'mlp']
algorithms = ['a2c', 'ppo', 'acktr']
reg_mode = ['always', 'converge', 'never']
reg_type = ['sum', 'reg', 'leaf']
param_init_func = {'uniform': lambda *x: 0.5 * torch.ones(*x),
                   'random': torch.rand,
                   'zero': torch.zeros}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #general
    parser.add_argument('--exp_root', type=str, default='tensorboards')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--run_suffix', type=str, default='')
    parser.add_argument('--save_model_interval', type=int, default=10)
    parser.add_argument('--run_log_dir_root', type=str, default='models')
    parser.add_argument('--plot_root', type=str, default='plots')
    parser.add_argument('--plot_tree', action='store_true')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--eval_interval', type=int, default=10,
                        help='eval interval, one eval per n updates (default: None)')
    #environment params
    parser.add_argument('--env_name', type=str, required=True)
    parser.add_argument('--num-processes', type=int, default=8,
                        help='how many training CPU processes to use (default: 16)')
    parser.add_argument('--converge_eps', type=float, default=1.0)
    #training length params
    parser.add_argument('--n_updates', type=int, default=int(200))

    #episode recording
    parser.add_argument('--record_video', action='store_true')
    parser.add_argument('--video_root', type=str, default='renderings')
    parser.add_argument('--video_dir_prefix', type=str, default='init')

    #regularization parameters
    parser.add_argument('--reg_init', type=float, default=1e1)
    parser.add_argument('--reg_mult', type=float, default=1.1)
    parser.add_argument('--reg_mode', type=str, choices=reg_mode, default='never')
    parser.add_argument('--reg_type', type=str, choices=reg_type, default='sum')
    parser.add_argument('--sig_alpha_init', type=float, default=1.0)
    parser.add_argument('--sig_alpha_mult', type=float, default=1.0)
    parser.add_argument('--sig_alpha_inc', type=float, default=10)
    #model parameters
    parser.add_argument('--model_type', type=str, choices=models)
    parser.add_argument('--ddt_depth', type=int, default=4)
    parser.add_argument('--param_init_func', type=str, default='random',
                        choices=list(param_init_func.keys()))

    #optimizer parameters
    parser.add_argument('--lr', type=float, default=1e-2)

    #rms prop params
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RMSprop optimizer apha (default: 0.99)')

    parser.add_argument('--cuda', action='store_true')

    #algorithm parameters
    parser.add_argument('--alg', type=str, choices=algorithms)
    parser.add_argument('--use-gae', action='store_true', default=False,
                        help='use generalized advantage estimation')
    parser.add_argument('--tau', type=float, default=0.95,
                        help='gae parameter (default: 0.95)')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')
    ## A2C algorithm parameters
    parser.add_argument('--n_steps', type=int, default=320,
                        help='number of forward steps in A2C (default: 5)')
    ## PPO algorithm parameters
    parser.add_argument('--ppo-epoch', type=int, default=10,
                        help='number of ppo epochs (default: 4)')
    parser.add_argument('--num-mini-batch', type=int, default=32,
                        help='number of batches for ppo (default: 32)')
    parser.add_argument('--clip-param', type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    args = parser.parse_args()
    pipeline = Pipeline(**vars(args))
    pipeline.run()