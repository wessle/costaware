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


def directories(path):
    """
    Returns a generator that iterates through all of the directories located in
    a specified path.

    Params
    ------
    path : str
        Path (relative or absolute) to search for directories

    Returns
    -------
    dirs : generator
        Iterates through the directories of the path
    """
    return filter(
        lambda name: os.path.isdir(os.path.join(path, name)),
        os.listdir(path)
    )


def get_data(root, drop=500, n_steps_to_skip=500):
    """
    Params
    ------
    root : str 
    drop : int (default=500)
    n_steps_to_skip : int (default=500)

    Returns
    -------
    data : pd.DataFrame
    """
    data = []
    for sd_name in directories(root):
        print(sd_name)
        sub_directory = os.path.join(root, sd_name)
        for experiment, ssd_name in enumerate(directories(sub_directory)):
            print(ssd_name)
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


def plot(data_dir, **kwargs):
    d = {
        'drop'            : 500,
        'n_steps_to_skip' : 10,
        'ext'             : '.png',
        'filename'        : 'comparison',
        'context'         : 'paper',
        'style'           : 'ticks',
        'palette'         : 'muted',
        'xlabel'          : 'steps',
        'ylabel'          : 'ratios',
        'title'           : 'title',
        'confidence'      : 95,

    }
    d.update(kwargs)
    kwargs = d

    data = get_data(
        data_dir,
        drop=kwargs['drop'], n_steps_to_skip=kwargs['n_steps_to_skip']
    )
    
    # save the figure
    fig, ax = Plotter(**kwargs).plot(data)
    fig.savefig(
        os.path.join(
            data_dir,
            kwargs['filename'] + '.' + kwargs['ext']
        ),
        bbox_inches='tight'
    ) 
    
    # save actual ratio data
    data.to_csv(
        os.path.join(
            data_dir,
            kwargs['filename'] + '.csv'
    ))
