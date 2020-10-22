import os
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from itertools import product

sns.set_context("paper")
sns.set_style("whitegrid")

Q = "Q"
AC = "AC"

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


def title_generator(agent):
    return f"{agent} portfolio agent"
    

def sci_formatter(number, pos=None):
    string = f"{number:1.1e}"
    mantissa = string[:3]
    exponent = int(string[-2:])
    return "$" + mantissa + "\\times 10^{" + str(exponent) + "}$"

def percent_formatter(number, pos=None):
    return f"{number:0.1f}%"


def plot_comparison_runs(fig, axes, q_learning, actor_critic, plot_runs=False, **kwargs):
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


def rho_experiment():
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


    fig, axes = plt.subplots(1, 1, figsize=config["figsize"])
    
    ql_idx = 0
    
    q_learning, actor_critic = load_experiment(
        config["top directory"], 
        config["filename"],
        ac_cut=config["cutoffs"]['ac'][0],
        ql_cut=config["cutoffs"]['ql'][0],
        load_ac=False
    )
    
    plot_comparison_runs(
        fig, 
        [axes], 
        q_learning, 
        None,
        plot_runs=True,alpha=0.,)

    for idx, label in zip([ql_idx], ["QL"]):
        axes.set_title(title_generator(label),
                       fontsize=10)
        axes.xaxis.set_major_formatter(FuncFormatter(sci_formatter))
        axes.yaxis.set_major_formatter(FuncFormatter(
            lambda number, pos=None: f"{number:1.1f}"
        ))
        axes.set_ylabel(config["ylabel"])
        axes.set_xlabel(config["xlabel"])
        axes.set_ylim(config["ylim"])
        plt.setp(axes.xaxis.get_majorticklabels(), 
                 fontsize=8,
                 rotation_mode="anchor",
                 rotation=10,
                 ha='right')
        sns.despine(ax=axes)

    plt.subplots_adjust(hspace=0.45)

    
    
    fig.savefig(f"{config['plot directory']}/{config['plot name']}.jpg", bbox_inches="tight")



def returns_experiment():
    config = {
        "top directory"     : "../data/portfolio_tests",
        "filename"          : "returns",
        "plot directory"    : "../plots",
        "plot name"         : "portfolio_returns",
        "cutoffs"           : {
            "ql": [100000],
            "ac": [100000],
        },
        "figsize"           : (8, 6),
        "xlabel"            : "Iterations",
        "ylabel"            : "Portfolio return after 1 year",
        "ylim"              : (-0.5, 0.5),
    }


    fig, axes = plt.subplots(
        2,
        1,
        figsize=config["figsize"],
    )
    
    ql_idx = 0
    ac_idx = 1
    axes[ql_idx].get_shared_y_axes().join(axes[ql_idx], axes[ac_idx])
    
    q_learning, actor_critic = load_experiment(
        config["top directory"], 
        config["filename"],
        ac_cut=config["cutoffs"]['ac'][0],
        ql_cut=config["cutoffs"]['ql'][0]
    )
    
    plot_comparison_runs(
        fig, 
        [axes[ql_idx], axes[ac_idx]], 
        q_learning, 
        actor_critic,
        plot_runs=False,
    )

    for idx, label in zip([ql_idx, ac_idx], ["QL", "AC"]):
        axes[idx].set_title(title_generator(label),
                            fontsize=10)
        axes[idx].xaxis.set_major_formatter(FuncFormatter(sci_formatter))
        axes[idx].yaxis.set_major_formatter(FuncFormatter(
            lambda number, pos=None: f"{number:1.1f}"
        ))
        axes[idx].set_ylabel(config["ylabel"])
        axes[idx].set_xlabel(config["xlabel"])
        axes[idx].set_ylim(config["ylim"])
        plt.setp(axes[idx].xaxis.get_majorticklabels(), 
                 fontsize=8,
                 rotation_mode="anchor",
                 rotation=10,
                 ha='right')
        sns.despine(ax=axes[idx])

    plt.subplots_adjust(hspace=0.45)

    
    
    fig.savefig(f"{config['plot directory']}/{config['plot name']}.jpg", bbox_inches="tight")

def returns_histogram():
    config = {
        "top directory"     : "../data/portfolio_tests",
        "filename"          : "end_values",
        "plot directory"    : "../plots",
        "plot name"         : "portfolio_histogram",
        "cutoffs"           : {
            "ql": [100000],
            "ac": [100000],
        },
        "figsize"           : (8, 6),
        "xlabel"            : "Portfolio value",
        "ylabel"            : "Frequency",
        "ylim"              : (0,0.0125),
    }


    fig, axes = plt.subplots(
        2,
        1,
        figsize=config["figsize"],
    )
    
    ql_idx = 0
    ac_idx = 1
    axes[ql_idx].get_shared_y_axes().join(axes[ql_idx], axes[ac_idx])
    
    q_learning, actor_critic = load_experiment(
        config["top directory"], 
        config["filename"],
        ac_cut=config["cutoffs"]['ac'][0],
        ql_cut=config["cutoffs"]['ql'][0]
    )
    
    # plot_comparison_runs(
    #     fig, 
    #     [axes[ql_idx], axes[ac_idx]], 
    #     q_learning, 
    #     actor_critic,
    #     plot_runs=False,)



    for idx, label, dataset in zip([ql_idx, ac_idx], ["QL", "AC"], [q_learning,
                                                                    q_learning]):
        axes[idx].hist(dataset.reshape(dataset.size), bins=50, density=True)

        axes[idx].set_title(title_generator(label),
                            fontsize=10)
        axes[idx].xaxis.set_major_formatter(FuncFormatter(sci_formatter))
        axes[idx].yaxis.set_major_formatter(FuncFormatter(
            lambda number, pos=None: f"{number:1.3f}"
        ))
        axes[idx].set_ylabel(config["ylabel"])
        axes[idx].set_xlabel(config["xlabel"])
        axes[idx].set_ylim(config["ylim"])
        plt.setp(axes[idx].xaxis.get_majorticklabels(), 
                 fontsize=8,
                 rotation_mode="anchor",
                 rotation=10,
                 ha='right')
        sns.despine(ax=axes[idx])

    plt.subplots_adjust(hspace=0.45)

    
    
    fig.savefig(f"{config['plot directory']}/{config['plot name']}.jpg", bbox_inches="tight")


rho_experiment()
returns_experiment()
# returns_histogram()
