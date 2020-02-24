import os
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from itertools import product

sns.set_context("paper")
sns.set_style("whitegrid")

def load_experiment(directory, filename, r, c, dim, ql_cut=800000, ac_cut=500000):
    def load_condition(subdir):
        split = subdir.split('-')
        return split[0] == f"r{r}c{c}" and split[1] == f"{dim}x{dim}"

    experiment = next(subdir for subdir in os.listdir(directory) if load_condition(subdir))

    q_learning = []
    actor_critic = []

    for exp_run in os.listdir(directory + "/" + experiment):
        ratios = np.load("/".join([directory, experiment, exp_run, f"{filename}.npy"]))
        actor = exp_run.split('-')[0]
        if actor  == "LinearAC":
            actor_critic.append(ratios[:ac_cut])
        elif actor == "TabularQ":
            q_learning.append(ratios[:ql_cut])

    return np.array(q_learning), np.array(actor_critic)


def title_generator(dim, r, c, agent):
    config = {
        "dimension": {
            5: "Small",
            10: "Medium",
            20: "Large",
        },
        "measure": {
            (2, 1): "Type-1",
            (3, 2): "Type-2",
            (4, 6): "Type-3",
        }
    }

    return " ".join([config["dimension"][dim], config["measure"][(r,c)], agent])


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


def experiment1():
    config = {
        "top directory"     : "data/alg_comparisons",
        "filename"          : "ratios",
        "reward-cost pairs" : [(2, 1), (3, 2), (4, 6)],
        "dimensions"        : [10, 20],
        "cutoffs"           : {
            "ql": [[100000, 800000, 500000], [60000, 3000000, 1000000]],
            "ac": [[100000, 400000, 500000], [250000,  200000,  100000]],
        },
        "figsize"           : (16, 12),
        "xlabel"            : "Iterations",
        "ylabel"            : "Ratio $\\rho$",
    }


    fig, axes = plt.subplots(
        2 * len(config["dimensions"]),
        len(config["reward-cost pairs"]),
        figsize=config["figsize"],
    )
    
    for d, dim in enumerate(config["dimensions"]):
        for i, (r, c) in enumerate(config["reward-cost pairs"]):
            ql_idx = (2*d, i)
            ac_idx = (2*d+1, i)
            axes[ql_idx].get_shared_y_axes().join(axes[ql_idx], axes[ac_idx])
    
            q_learning, actor_critic = load_experiment(
                config["top directory"], 
                config["filename"],
                r, c, dim,
                ac_cut=config["cutoffs"]['ac'][d][i],
                ql_cut=config["cutoffs"]['ql'][d][i]
            )
    
            plot_comparison_runs(
                fig, 
                [axes[ql_idx], axes[ac_idx]], 
                q_learning, 
                actor_critic,
                alpha=0.,
                plot_runs=True,)

            for idx, label in zip([ql_idx, ac_idx], ["QL", "AC"]):
                axes[idx].set_title(title_generator(dim, r, c, label),
                                    fontsize=10)
                axes[idx].xaxis.set_major_formatter(FuncFormatter(sci_formatter))
                axes[idx].yaxis.set_major_formatter(FuncFormatter(
                    lambda number, pos=None: f"{number:1.1f}"
                ))
                axes[idx].set_ylabel(config["ylabel"])
                axes[idx].set_xlabel(config["xlabel"])
                plt.setp(axes[idx].xaxis.get_majorticklabels(), 
                         fontsize=8,
                         rotation_mode="anchor",
                         rotation=10,
                         ha='right')
                sns.despine(ax=axes[idx])

            plt.subplots_adjust(hspace=0.45)

    # for i in range(2):
    #     axes[0,i].set_ylim(config["ylim"][i])
    
    
    fig.savefig("plots/experiment1.jpg", bbox_inches="tight")

def experiment2():
    config = {
        "top directory"     : "data/different_inits",
        "filename"          : "ratios",
        "reward-cost pairs" : [(2, 1), (3, 2)],
        "dimensions"        : [10],
        "cutoffs"           : {
            "ql": [[40000, 800000]],
            "ac": [[120000, 120000]],
        },
        "figsize"           : (7, 5),
        "ylim"              : [(0, 23), (0, 10)],
        "ylabel"            : "Ratio $\\rho$",
        "xlabel"            : "Iterations",
    }

    fig, axes = plt.subplots(
        2 * len(config["dimensions"]),
        len(config["reward-cost pairs"]),
        figsize=config["figsize"],
        sharey='col',
    )
    
    for d, dim in enumerate(config["dimensions"]):
        for i, (r, c) in enumerate(config["reward-cost pairs"]):
            ql_idx = (2*d, i)
            ac_idx = (2*d+1, i)
    
            q_learning, actor_critic = load_experiment(
                config["top directory"], 
                config["filename"],
                r, c, dim,
                ac_cut=config["cutoffs"]['ac'][d][i],
                ql_cut=config["cutoffs"]['ql'][d][i]
            )
    
            plot_comparison_runs(
                fig, 
                [axes[ql_idx], axes[ac_idx]], 
                q_learning, 
                actor_critic,
                plot_runs=True,
                alpha=0.
            )
    
            for idx, label in zip([ql_idx, ac_idx], ["QL", "AC"]):
                axes[idx].set_title(title_generator(dim, r, c, label),
                                    fontsize=10)
                axes[idx].xaxis.set_major_formatter(FuncFormatter(sci_formatter))
                axes[idx].yaxis.set_major_formatter(FuncFormatter(
                    lambda number, pos=None: f"{number:1.0f}"
                ))
                axes[idx].set_ylabel(config["ylabel"])
                axes[idx].set_xlabel(config["xlabel"])
                plt.setp(axes[idx].xaxis.get_majorticklabels(), 
                         fontsize=8,
                         rotation_mode="anchor",
                         rotation=10,
                         ha='right')
                sns.despine(ax=axes[idx])
            plt.subplots_adjust(hspace=0.45, wspace=0.2)

    for i, j in product(range(2), repeat=2):
        axes[i,i].set_ylim(config["ylim"][i])
    

    
    fig.savefig("plots/experiment2.jpg", bbox_inches="tight")


def experiment3():
    config = {
        "top directory"     : "data/mc_control",
        "filename"          : "percent_time_in_goal",
        "reward-cost pairs" : [(4, 6)],
        "dimensions"        : [5,10],
        "cutoffs"           : {
            "ql": [[200000], [200000]],
            "ac": [[200000], [200000]],
        },
        "figsize"           : (7, 5),
        "ylim"              : [(0, 40), (0, 20)],
        "ylabel"            : "Percent spent in state $0$",
        "xlabel"            : "Iterations",
    }

    fig, axes = plt.subplots(
        len(config["dimensions"]),
        2,
        figsize=config["figsize"],
        sharey='col'
    )
    
    for d, dim in enumerate(config["dimensions"]):
        for i, (r, c) in enumerate(config["reward-cost pairs"]):
            ql_idx = (i, d)
            ac_idx = (i+1, d)
    
            q_learning, actor_critic = load_experiment(
                config["top directory"], 
                config["filename"],
                r, c, dim,
                ac_cut=config["cutoffs"]['ac'][d][i],
                ql_cut=config["cutoffs"]['ql'][d][i]
            )
    
            plot_comparison_runs(
                fig, 
                [axes[ql_idx], axes[ac_idx]], 
                q_learning, 
                actor_critic,
                plot_runs=True,
                alpha=0.
            )
    
            for idx, label in zip([ql_idx, ac_idx], ["QL", "AC"]):
                axes[idx].set_title(title_generator(dim, r, c, label),
                                    fontsize=10)
                axes[idx].xaxis.set_major_formatter(FuncFormatter(sci_formatter))
                axes[idx].yaxis.set_major_formatter(FuncFormatter(percent_formatter))
                axes[idx].set_ylabel(config["ylabel"])
                axes[idx].set_xlabel(config["xlabel"])
                plt.setp(axes[idx].xaxis.get_majorticklabels(), 
                         fontsize=8,
                         rotation_mode="anchor",
                         rotation=10,
                         ha='right')
                sns.despine(ax=axes[idx])
            plt.subplots_adjust(hspace=0.5, wspace=0.4)

    for i in range(2):
        axes[0,i].set_ylim(config["ylim"][i])
    
    
    fig.savefig("plots/experiment3.jpg", bbox_inches="tight")


experiment1()
experiment2()
experiment3()
