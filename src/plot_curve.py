import argparse
import numpy as np
import random
import matplotlib
import seaborn as sns; sns.set()
matplotlib.use('TkAgg') # Can change to 'Agg' for non-interactive mode

import matplotlib.pyplot as plt

import plot_util as pu

plt.rcParams['svg.fonttype'] = 'svgfont'
# plt.rcParams.update({'font.size': 22})
font_size = 28

matplotlib.rc('xtick', labelsize=font_size) 
matplotlib.rc('ytick', labelsize=font_size) 


def argsparser():
    parser = argparse.ArgumentParser("Plotting lines")
    parser.add_argument('--env_id', help='env', default='DeepMimic')
    parser.add_argument('--legend', help='legend', type=int, default=1)
    return parser.parse_args()

def main(args):
    env_name = args.env_id.split("_")[0]
    dir_path = 'log_tmp/%s'%(args.env_id)
    results = pu.load_results(dir_path)

    pu.plot_results(results, 
                    average_group=True, 
                    split_fn=lambda _: '',  
                    # disables splitting; all curves end up on the same panel
                    figsize=(8, 8),
                    # legend_outside=True,
                    # num_timesteps=args.num_timesteps,
                    font_size=font_size,
                    shaded_std=False, 
                    legend=args.legend)
    # plt.grid(which='major')
    plt.xlabel('Number of interactions (%s)'%(args.env_id), fontsize=font_size)
    plt.ylabel('Rewards', fontsize=font_size)
    plt.tight_layout()
    params = {'legend.fontsize': font_size}
    plt.rcParams.update(params)
    fig_name = "figures/%s-reward.svg"%(args.env_id)
    plt.savefig(fig_name)
    # plt.show()

if __name__ == '__main__':
    args = argsparser()
    main(args)

'''
import numpy as np
import matplotlib
matplotlib.use('TkAgg') # Can change to 'Agg' for non-interactive mode

import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'

import seaborn as sns; sns.set()

from baselines.common import plot_util

X_TIMESTEPS = 'timesteps'
X_EPISODES = 'episodes'
X_WALLTIME = 'walltime_hrs'
Y_REWARD = 'reward'
Y_TIMESTEPS = 'timesteps'
POSSIBLE_X_AXES = [X_TIMESTEPS, X_EPISODES, X_WALLTIME]
EPISODES_WINDOW = 100
COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink',
        'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise',
        'darkgreen', 'tan', 'salmon', 'gold', 'darkred', 'darkblue']

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def window_func(x, y, window, func):
    yw = rolling_window(y, window)
    yw_func = func(yw, axis=-1)
    return x[window-1:], yw_func

def ts2xy(ts, xaxis, yaxis):
    if xaxis == X_TIMESTEPS:
        x = np.cumsum(ts.l.values)
    elif xaxis == X_EPISODES:
        x = np.arange(len(ts))
    elif xaxis == X_WALLTIME:
        x = ts.t.values / 3600.
    else:
        raise NotImplementedError
    if yaxis == Y_REWARD:
        y = ts.r.values
    elif yaxis == Y_TIMESTEPS:
        y = ts.l.values
    else:
        raise NotImplementedError
    return x, y

def plot_curves(xy_list, xaxis, yaxis, title):
    fig = plt.figure(figsize=(8,2))
    maxx = max(xy[0][-1] for xy in xy_list)
    minx = 0
    for (i, (x, y)) in enumerate(xy_list):
        color = COLORS[i % len(COLORS)]
        plt.scatter(x, y, s=2)
        x, y_mean = window_func(x, y, EPISODES_WINDOW, np.mean) #So returns average of last EPISODE_WINDOW episodes
        plt.plot(x, y_mean, color=color)
    plt.xlim(minx, maxx)
    plt.title(title)
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    plt.tight_layout()
    fig.canvas.mpl_connect('resize_event', lambda event: plt.tight_layout())
    plt.grid(True)


def split_by_task(taskpath):
    return taskpath['dirname'].split('/')[-1].split('-')[0]

def plot_results(dirs, num_timesteps=10e4, xaxis=X_TIMESTEPS, yaxis=Y_REWARD, title='', split_fn=split_by_task):
    results = plot_util.load_results(dirs)
    plot_util.plot_results(results, 
                           # y_fn=lambda r: ts2xy(r['monitor'], xaxis, yaxis), 
                           split_fn=lambda _: '', 
                           shaded_std=False,
                           # resample=int(1e6),
                           average_group=True)

# Example usage in jupyter-notebook
# from baselines.results_plotter import plot_results
# %matplotlib inline
# plot_results("./log")
# Here ./log is a directory containing the monitor.csv files

def main():
    import argparse
    import os
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num_timesteps', type=int, default=int(10e4))
    parser.add_argument('--env_id', help='env', default='HalfCheetah')
    parser.add_argument('--xaxis', help = 'Varible on X-axis', default = X_TIMESTEPS)
    parser.add_argument('--yaxis', help = 'Varible on Y-axis', default = Y_REWARD)
    parser.add_argument('--task_name', help = 'Title of plot', default = 'Breakout')
    args = parser.parse_args()
    if args.env_id == 'CartPole':
        args.dirs  = ['log_trpo_cartpole/%s'%(args.env_id)]
    else:
        args.dirs = ['log_trpo_mujoco/%s'%(args.env_id)]
    plot_results(args.dirs, args.num_timesteps, args.xaxis, args.yaxis, args.task_name)
    plt.show()

if __name__ == '__main__':
    main()
'''


'''
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid")

def argsparser():
    parser = argparse.ArgumentParser("Plotting lines")
    parser.add_argument('--csv_path', help='csv path', default='log_trpo_mujoco/trpo_stochastic.HalfCheetah.g_step_3.d_step_1.policy_entcoeff_0.adversary_entcoeff_0.001.seed_0/progress.csv')
    return parser.parse_args()

def main(args):
    data = pd.read_csv(args.csv_path)

    # Plot the responses for different events and regions
    # sns.tsplot(value='signal', time='timepoint', condition='subject', linestyle="--", marker='s', data=fmri)

    sns.lineplot(x="timepoint", y="EpRewMean", hue="region", style="event", data=fmri)

    plt.show()

if __name__ == '__main__':
    args = argsparser()
    main(args)
'''