from collections import OrderedDict
import numpy as np
from typing import List, Tuple

class Portfolio:
    """
    """

    def __init__(self, assets: List['Asset'], weights: List[float], principal: float) -> None:
        """
        """
        self.__assets    = assets
        self.__weights   = weights
        self.__principal = principal
        self.__value     = principal
        self.__shares    = None

    @property
    def assets(self) -> List['Asset']:
        return self.__assets

    @property 
    def weights(self) -> List[float]:
        return self.__weights

    @property
    def principal(self) -> float:
        return self.__principal

    @property
    def value(self) -> float:
        return self.__value

    @property
    def shares(self) -> List[float]:
        return self.__shares

    @principal.setter
    def principal(self, new_principal: float) -> None:
        self.__principal = new_principal

    @weights.setter
    def weights(self, new_weights: List[float]) -> None:
        self.__weights = new_weights

    def __len__(self) -> int:
        return len(self.__assets)

    def reset(self) -> float:
        """
        Resets each asset in the portfolio
        """
        for asset in self.assets:
            asset.reset()
        self.__shares = [weight * self.value / asset.price for asset, weight in zip(self.assets, self.weights)]

    def step(self) -> Tuple[float, List['Asset'], List[float]]:
        """
        """
        _             = [asset.step() for asset in self.assets]
        self.__value  = sum(num_shares * asset.price for num_shares, asset in zip(self.shares, self.assets))
        self.__shares = [weight * self.value / asset.price for asset, weight in zip(self.assets, self.weights)]

        return self.summary

    @property
    def summary(self):
        """
        Get a snapshot of the current state of the portfolio without updating
        the state at all.
        """
        state = [self.value, *self.shares]
        for asset in self.assets:
            state += [*asset.summary()]

        return np.array(state)
