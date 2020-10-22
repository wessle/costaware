import wesutils

import numpy as np
import matplotlib.pyplot as plt

CONFIG_FILE = 'config.yml'
AC_SUFFIX   = 'LinearAC_ratios.npy'
TQ_SUFFIX   = 'TabularQ_ratios.npy'
PLOT_FILE   = 'comparison.jpg'

config = wesutils.load_config(CONFIG_FILE)

n_axes = len(config['num_states'])

fig, axes = plt.subplots(1, n_axes)

for i, ax in enumerate(axes):
    n_state  = config['num_states'][i]
    n_action = config['num_states'][i]

    PREFIX = f's{n_state}a{n_action}'

    ac_ratios = np.load("_".join([PREFIX, AC_SUFFIX]))
    tq_ratios = np.load("_".join([PREFIX, TQ_SUFFIX]))
    steps     = np.arange(ac_ratios.size)
    
    ax.plot(steps, ac_ratios, label='LinearAC')
    ax.plot(steps, tq_ratios, label='TabularQ')
    
    ax.set_xlabel('steps')
    ax.set_ylabel('ratio')
    ax.set_title('Comparison of ratios')
    ax.legend()

fig.savefig(PLOT_FILE, bbox_inches='tight')
