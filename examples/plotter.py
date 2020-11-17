import numpy as np
import pandas as pd
from numpy.random import default_rng
from scipy import stats
from scipy.special import expit
import matplotlib.pyplot as plt
import seaborn as sns

rng = default_rng()

def simulated_noise(shape):
    return rng.standard_normal(shape)

def simulated_output(x, mean, capacity, rate):
    x = (x - mean) / x.size
    signal = capacity * expit(rate * x)
    noise = simulated_noise(x.shape) * np.exp(-x)
    return signal + noise


confidence_level = 95
n_steps = 100000
n_sims = 10
n_series = 2

params = {
    0: (n_steps / 2, 1., 10),
    1: (3 * n_steps / 4, 2., 10)
}

def get_data(downsample=None):
    """
    downsample : int
        reduce dataframe by factor of downsample
    """
    steps = np.arange(n_steps)
    ds = n_steps // downsample
    
    data = pd.DataFrame([
        (step, x, i, series) \
        for series in range(n_series) \
        for i in range(n_sims) \
        for step, x in zip(steps, simulated_output(steps, *params[series]))
    ], columns=['step', 'ratio', 'sim', 'series'])
    data = data[data['step'] % ds == 0]
    
    return data


class Plotter:

    def __init__(self, **kwargs):
        defaults = {
            'context' : 'paper',
            'style'   : 'ticks',
            'palette' : 'muted',
            'xlabel'  : 'steps',
            'ylabel'  : 'ratios',
            'title'   : 'title',
        }
        defaults.update(kwargs)
        self.__dict__.update(defaults)

    def plot(self, data):
        sns.set_context(self.context)
        sns.set_style(self.style)
        sns.set_palette(self.palette)

        fig, ax = plt.subplots()
        sns.lineplot(data=data, x='step', y='ratio', hue='series',
                     ci=confidence_level)

        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.set_title(self.title)
        ax.legend()
        sns.despine(fig=fig, ax=ax)

        return fig, ax



data = get_data(downsample=150)
fig, ax = Plotter().plot(data)
fig.savefig('test.png', bbox_inches='tight')
data.to_html('test.html')
