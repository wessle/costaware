import pickle
import os

import main.core.portfolio as portfolio
import main.core.asset as asset


def make_portfolio(init_prices,
                   mean_returns,
                   stdev_returns,
                   init_weights,
                   init_principal,
                   asset_class='SimpleAsset'):
    """
    Take lists of asset parameters, then return a corresponding portfolio.
    """

    asset_params = zip(init_prices, mean_returns, stdev_returns)
    assets = [eval('asset.' + asset_class)(*param) for param in asset_params]
    return portfolio.Portfolio(assets, init_weights, init_principal)


def plot_average_fig(pickle_name, plot_filename):
    """
    Read in pickle files (all with the same name) from all
    subdirectories in the current directory containing lists of returns
    and plot their average.
    """

    for _, _, files in os.walk('.'):
        returns = [load_object(file) for file in file if file == pickle_name]
