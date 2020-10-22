import re
import numpy as np
import matplotlib.pyplot as plt

directory = 'data/TabularQ-MDP-2020-10-08_18:44:07/ratios.npy'

def simple_moving_average(series, width):
    """
    Returns a simple moving average of a 1-dimensional array of data with
    arbitrary width (i.e., level of smoothing)
    """
    return np.array([
        series[i:series.size-width+i+1] for i in range(width)
    ]).mean(axis=0)

def load_experiment(directory, filename, ql_cut=800000, ac_cut=500000,
                    load_ac=True):
    q_learning = []
    actor_critic = []

    for exp_run in os.listdir(directory):
        actor = exp_run.split('-')[0]
        if actor == Q:
            ratios = np.load("/".join([directory, exp_run, f"{filename}.npy"]))
            q_learning.append(ratios[:ql_cut])
            print("Loading " + "/".join([directory, exp_run, f"{filename}.npy"]))
        elif actor == AC and load_ac:
            ratios = np.load("/".join([directory, exp_run, f"{filename}.npy"]))
            print("Loading " + "/".join([directory, exp_run, f"{filename}.npy"]))
            actor_critic.append(ratios[:ac_cut])

    return np.array(q_learning), np.array(actor_critic)


class MDPComparisonPlotter:

    def __init__(self, config):
        self.config = config
        self.fig, self.axes = plt.subplots(1, 1, figsize=self.config["figsize"])

        self.q_learning, self.actor_critic = load_experiment(
            self.config["top directory"], 
            self.config["filename"],
            ac_cut=self.config["cutoffs"]['ac'][0],
            ql_cut=self.config["cutoffs"]['ql'][0],
            load_ac=False
        )

        # regexp for parsing mantissa and exponent from an e-formatted string
        self.__sci_re = re.compile('(\d+\.\d*)e\+(\d+)')
        self.__prec = 1

    def title(self, agent):
        return f"{agent} portfolio agent"
        
    
    def _sci_fmttr(self, number, pos=None):
        """
        Returns a LaTeX-formatted string of the given number in scientific
        notation (base 10).
        """
        mantissa, exponent = self.__sci_re.match(
            "{num:1.{prec}e}".format(num=number, prec=self.__prec)
        ).groups()

        return "${man} \\times 10^{{{exp}}}$".format(
            man=mantissa, exp=str(int(exponent))
        )

    def _flt_fmttr(self, number, pos=None):
        """
        Returns a float-formatted string of the given number
        """
        return "{num:1.{prec}f}".format(num=number, prec=self.__prec)
    
    def plot_comparison(fig, axes, q_learning, actor_critic, plot_runs=False, **kwargs):
        default_kwargs = {
            'colors': ['r', 'b'],
            'alpha': 0.15,
            'linewidth': 0.,
        }
        kwargs = {a: kwargs.get(a, b) for a, b in default_kwargs.items()}
    
        for i, dataset in enumerate([q_learning, actor_critic]):
            if dataset is None:
                break
            mean = dataset.mean(axis=0)
            std  = dataset.std(axis=0)
            steps = np.arange(mean.size)
    
            if plot_runs:
                for run in dataset:
                    axes[i].plot(steps, run, color='000000', alpha=0.2)
    
            axes[i].plot(steps, mean, color=kwargs['colors'][i])
            axes[i].fill_between(
                steps, mean - std, mean + std, 
                linewidth=kwargs['linewidth'],
                alpha=kwargs['alpha'],
                color=kwargs['colors'][i]
            )
    
            axes[i].fill_between(
                steps, mean - 2 * std, mean + 2 * std, 
                linewidth=kwargs['linewidth'],
                alpha=kwargs['alpha'],
                color=kwargs['colors'][i],
            )
    
        return fig, axes

    def rho_experiment(self):
        config = {
            "top directory"     : "../data/portfolio_tests",
            "filename"          : "rhos",
            "plot directory"    : "../plots",
            "plot name"         : "portfolio_rhos",
            "cutoffs"           : {
                "ql": [100000],
                "ac": [100000],
            },
            "figsize"           : (4, 3),
            "xlabel"            : "Iterations",
            "ylabel"            : "Rho $\\rho$",
            "ylim"              : (0,6),
        }
    
        ql_idx = 0
        
        self.plot_comparison_runs(self.fig, [self.axes], self.q_learning, self.actor_critic,
            plot_runs=True,alpha=0.,)
    
        for idx, label in zip([ql_idx], ["QL"]):
            self.axes.set_title(self.title(label), fontsize=10)
            self.axes.xaxis.set_major_formatter(FuncFormatter(self._sci_fmttr))
            self.axes.yaxis.set_major_formatter(FuncFormatter(self._flt_fmttr))
            self.axes.set_ylabel(self.config["ylabel"])
            self.axes.set_xlabel(self.config["xlabel"])
            self.axes.set_ylim(self.config["ylim"])
            plt.setp(self.axes.xaxis.get_majorticklabels(), 
                     fontsize=8,
                     rotation_mode="anchor",
                     rotation=10,
                     ha='right')
            sns.despine(ax=self.axes)
    
        plt.subplots_adjust(hspace=0.45)

    def save(self):
        self.fig.savefig(
            f"{self.config['plot directory']}/{self.config['plot name']}.jpg", 
            bbox_inches="tight"
        )

