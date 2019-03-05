import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import os
import seaborn as sns
import argparse
import numpy as np
from tensorboard.backend.event_processing import event_accumulator as ea

sns.set(style="darkgrid")

scalar_tags = ['Reward',
               'Mean_History_Reward',
               'Regterm',
               'Converged',
               'Std_History_Reward',
               'Dist_Entropy',
               'Inst_Mean_Eval_Reward']

reward_tag = scalar_tags[0]
mean_reward_tag = scalar_tags[1]
std_history_tag = scalar_tags[4]
mean_eval_reward_tag = scalar_tags[6]


class Plotter:

    def __init__(self, **kwargs):

        for k, v in kwargs.items():
            setattr(self, k, v)
        self.exp_root = os.path.normpath(self.exp_root)
        self.env = os.path.basename(self.exp_root)
        self.models = os.listdir(self.exp_root)
        self.save_dir = os.path.join('plots', self.env)
        os.makedirs(self.save_dir, exist_ok=True)

    def get_raw_list(self, log_path, tag_extracted):
        acc = ea.EventAccumulator(log_path,
                                  size_guidance={ea.SCALARS: 0},
                                  purge_orphaned_data=False)
        acc.Reload()
        # only support scalar now
        # scalar_list = acc.Tags()['scalars']
        x = np.array([int(s.step) for s in acc.Scalars(tag_extracted)])
        y = np.array([s.value for s in acc.Scalars(tag_extracted)])
        return x, y

    def get_model_frame(self, model_name):
        model_root = os.path.join(self.exp_root, model_name)
        exp_dirs = [os.path.join(model_root, exp)
                    for exp in os.listdir(model_root)]
        xs, ys = list(), list()
        min_length = float('Inf')
        for exp_dir in exp_dirs:
            try:
                x, y = self.get_raw_list(exp_dir, self.plt_tag)
                xs.append(x)
                ys.append(y)
                if x.shape[0] < min_length:
                    min_length = x.shape[0]
            except Exception:
                print('Failed To Load Experiment: {}'.format(exp_dir))
        if len(xs) == 0:
            return None
        xs = np.concatenate([xs[0][:min_length] for x in xs])
        ys = np.concatenate([y[:min_length] for y in ys])
        model = np.array([model_name for i in range(xs.shape[0])])
        data = np.stack([xs, ys, model], axis=1)

        frame = pd.DataFrame(data=data, columns=['samples', 'reward', 'model'])
        frame['samples'] = frame['samples'].astype(np.int64)
        frame['reward'] = frame['reward'].astype(np.float64)
        return frame

    def plot(self):
        frames = list()

        for model in self.models:
            m_frame = self.get_model_frame(model)
            if m_frame is not None:
                frames.append(m_frame)
        max_samples = min([f.max(0)['samples'] for f in frames])
        frame = pd.concat([f[f['samples'] <= max_samples] for f in frames])
        sns.lineplot(x='samples', y='reward', hue='model', data=frame)
        plt.title(self.env)
        plt.savefig(os.path.join(self.save_dir, '{}.svg'.format(self.plt_tag)))


tag_choices = [reward_tag, mean_reward_tag, std_history_tag, mean_eval_reward_tag]
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_root', type=str, default='tensorboards')
    parser.add_argument('--plt_tag', type=str, choices=tag_choices,
                        default=mean_eval_reward_tag)
    args = parser.parse_args()
    plotter = Plotter(**vars(args))
    plotter.plot()
