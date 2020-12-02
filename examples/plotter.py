import argparse
import os
import yaml
import numpy as np
import pandas as pd
from numpy.random import default_rng
from scipy import stats
from scipy.special import expit
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc



parser = argparse.ArgumentParser('python plotter.py')
parser.add_argument('--data_dir', type=str,
                    default=None,
                    help='Directory to store trial data in')

args = parser.parse_args()

def directories(path):
    return filter(
        lambda name: os.path.isdir(os.path.join(path, name)),
        os.listdir(path)
    )


def get_data(root=args.data_dir, drop=500, n_steps_to_skip=500):
    data = []
    for sd_name in directories(root):
        sub_directory = os.path.join(root, sd_name)
        for experiment, ssd_name in enumerate(directories(sub_directory)):
            subsub_directory = os.path.join(sub_directory, ssd_name)
            ratios = np.load(os.path.join(subsub_directory, 'ratios.npy'))
            config = yaml.safe_load(
                open(os.path.join(subsub_directory, 'config.yml'), 'r')
            )
            agent = config['agent_config']['class']
            trial_set_name = sd_name  # config['trial_set_name']

            for step, ratio in enumerate(ratios):
                    data.append((
                        step, ratio, experiment, agent, trial_set_name
                    ))

    data = pd.DataFrame(data, columns=['step', 'ratio', 'experiment', 'agent',
                                       'trial_set_name'])
    data = data[data['step'] % drop == 0]
    data = data[data['step'] >= n_steps_to_skip]

    return data
                    

class Plotter:
    """
    TODOs:
    ------
    * Can we toggle showing AC only or Q only?`
    """

    def __init__(self, **kwargs):
        defaults = {
            'context'    : 'paper',
            'style'      : 'ticks',
            'palette'    : 'muted',
            'xlabel'     : 'steps',
            'ylabel'     : 'ratios',
            'title'      : 'title',
            'confidence' : 95,
        }
        defaults.update(kwargs)
        self.__dict__.update(defaults)

    def plot(self, data):
        sns.set_context(self.context)
        sns.set_style(self.style)
        sns.set_palette(self.palette)

        fig, ax = plt.subplots(figsize=(8,6))
        sns.lineplot(data=data, x='step', y='ratio', hue='trial_set_name',
                     ci=self.confidence)

        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.set_title(self.title)
        ax.legend()
        sns.despine(fig=fig, ax=ax)

        return fig, ax

OUTPUT_NAME = 'comparison'
PLOT_FMT = '.png'

data = get_data(drop=150)

fig, ax = Plotter(confidence=90).plot(data)
fig.savefig(os.path.join(args.data_dir, OUTPUT_NAME + PLOT_FMT), bbox_inches='tight') 

data.to_csv(os.path.join(args.data_dir, OUTPUT_NAME + '.csv'))
