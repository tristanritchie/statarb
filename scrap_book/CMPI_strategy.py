"""
Rules:

"""
from c_vine_model import CVineModel
from pyvinecopulib import to_pseudo_obs
import pandas as pd
import matplotlib.pyplot as plt


class CMPI:
    DEFAULT_FAMILY = [1, 2, 3, 4, 5, 6, 7, 9, 10, 13, 16, 20, 23, 24, 26, 27, 29, 30, 33, 34, 36, 39, 40, 
                    104, 114, 124, 134, 204, 214, 224, 234]
    def __init__(self) -> None:
        self.cvm = CVineModel()
        self.family_set = None
        self.training_returns = None

    def pseudo_obs(self, data: pd.DataFrame):
        """
        Calculate quantiles from stock returns (ECDF)
        """
        columns = list(data)
        quantiles = to_pseudo_obs(data)
        return pd.DataFrame(quantiles, columns=columns) 

    def init_copula_model(self, training_returns: pd.DataFrame, family_set: list = None):
        self.training_returns = training_returns
        # Get quantiles data
        quantiles = self.pseudo_obs(training_returns)
        # Fit copula
        self.cvm.fit(quantiles, CMPI.DEFAULT_FAMILY)
        #TODO: Implement goodness-of-fit checks here

    def calculate_bollinger_bands(self, series: pd.Series, window_size: int, num_std: int):
        rolling = series.rolling(window_size)
        bband = rolling.mean() + rolling.std(ddof=0) * num_std
        return bband

    def generate_trading_signals(self, testing_returns: pd.DataFrame) -> pd.Series:
        trading_periods = testing_returns.shape[0]
        copula_fit_length = self.training_returns.shape[0]
        
        all_returns = pd.concat([self.training_returns, testing_returns], ignore_index=True)

        cmpi_list = [0]
        start = 0
        stop = copula_fit_length
        while stop < trading_periods + copula_fit_length:
            quantiles = self.pseudo_obs(all_returns[start:stop])
            self.cvm.fit(quantiles, self.family_set)
            # get misspricing index
            mpi = self.cvm.predict(quantiles[-1:])
            # de-mean misspricing index and cummulatively sum to get CMPI
            cmpi_list.append(cmpi_list[-1] + (mpi.values[0] - 0.5))
            # increment trading period
            start += 1
            stop += 1

        cmpi = pd.Series(cmpi_list)
        return cmpi
       

        