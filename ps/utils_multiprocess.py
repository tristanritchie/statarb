import numpy as np
import pandas as pd
import multiprocessing as mp
from itertools import repeat

from ps.ps_utils import get_sum_correlations, multivariate_rho, diagonal_measure, extremal_measure


def _traditional_correlation_loop(corr_matrix: pd.DataFrame, molecule: list) -> pd.DataFrame:
    """
    Calculates sum of correlations for each quadruple in the molecule.
    :param corr_matrix: (pd.DataFrame) Correlation Matrix
    :param molecule: (list) Indices of quadruples
    :return: (pd.DataFrame) Sum of correlations for each quadruple
    """
    results = []
    for quadruple in molecule:
        results.append((quadruple, get_sum_correlations(corr_matrix, quadruple)))

    return pd.DataFrame(results, columns=['quadruple', 'result'])


def run_traditional_correlation_calcs(corr_matrix: pd.DataFrame, quadruples: list, num_threads: int = 8,
                                      verbose: bool = True) -> pd.DataFrame:
    """
    This function is the multi threading wrapper that supplies _traditional_correlation_loop with quadruples.
    :param corr_matrix: (pd.DataFrame) Correlation Matrix
    :param quadruples: (list)  list of quadruples
    :param num_threads: (int) Number of cores to use
    :param verbose: (bool) Flag to report progress on asynch jobs
    :return: (pd.DataFrame) Quadruple with highest sum of correlations
    """

    chunk_size = len(quadruples) // num_threads
    quadruple_chunks = [quadruples[i:i + chunk_size] for i in range(0, len(quadruples), chunk_size)]

    with mp.Pool(num_threads) as p:
        results_list = p.starmap(_traditional_correlation_loop, zip(repeat(corr_matrix), quadruple_chunks))

    results = pd.concat(results_list)

    return results.iloc[results['result'].argmax()]


def _extended_correlation_loop(u_matrix: pd.DataFrame, molecule: list) -> pd.DataFrame:
    """
    Calculates multivariate Spearman's for each quadruple in the molecule.
    :param u_matrix: (pd.DataFrame) ranked returns
    :param molecule: (list) Indices of quadruples
    :return: (pd.DataFrame) Mean of the three estimators of multivariate rho for each quadruple
    """
    results = []
    for quadruple in molecule:
        results.append((quadruple, multivariate_rho(u_matrix[quadruple])))

    return pd.DataFrame(results, columns=['quadruple', 'result'])


def run_extended_correlation_calcs(u: pd.DataFrame, quadruples: list, num_threads: int = 8,
                                   verbose: bool = True) -> pd.DataFrame:
    """
    This function is the multi threading wrapper that supplies _extended_correlation_loop with quadruples.
    :param u: (pd.DataFrame) ranked returns
    :param quadruples: (list)  list of quadruples
    :param num_threads: (int) Number of cores to use
    :param verbose: (bool) Flag to report progress on asynch jobs
    :return: (pd.DataFrame) Quadruple with highest multivariate correlation
    """
    chunk_size = len(quadruples) // num_threads
    quadruple_chunks = [quadruples[i:i + chunk_size] for i in range(0, len(quadruples), chunk_size)]

    with mp.Pool(num_threads) as p:
        results_list = p.starmap(_extended_correlation_loop, zip(repeat(u), quadruple_chunks))

    results = pd.concat(results_list)

    return results.iloc[results['result'].argmax()]


def _diagonal_measure_loop(ranked_returns: pd.DataFrame, molecule: list) -> pd.DataFrame:
    """
    Calculates diagonal measure for each quadruple in the molecule.
    :param ranked_returns: (pd.DataFrame) ranked returns
    :param molecule: (list) Indices of quadruples
    :return: (pd.DataFrame) total euclidean distance for each quadruple
    """
    results = []
    for quadruple in molecule:
        results.append((quadruple, diagonal_measure(ranked_returns[quadruple])))

    return pd.DataFrame(results, columns=['quadruple', 'result'])


def run_diagonal_measure_calcs(ranked_returns: pd.DataFrame, quadruples: list, num_threads: int = 8,
                               verbose: bool = True) -> pd.DataFrame:
    """
    This function is the multi threading wrapper that supplies _diagonal_measure_loop with quadruples.
    :param ranked_returns: (pd.DataFrame) ranked returns
    :param quadruples: (list)  list of quadruples
    :param num_threads: (int) Number of cores to use
    :param verbose: (bool) Flag to report progress on asynch jobs
    :return: (pd.DataFrame) Quadruple with smallest diagonal measure
    """

    chunk_size = len(quadruples) // num_threads
    quadruple_chunks = [quadruples[i:i + chunk_size] for i in range(0, len(quadruples), chunk_size)]

    with mp.Pool(num_threads) as p:
        results_list = p.starmap(_diagonal_measure_loop, zip(repeat(ranked_returns), quadruple_chunks))

    results = pd.concat(results_list)

    return results.iloc[results['result'].argmin()]


def _extremal_measure_loop(ranked_returns: pd.DataFrame, co_variance_matrix: np.array, molecule: list) -> pd.DataFrame:
    """
    Calculates extremal measure for each quadruple in the molecule.
    :param co_variance_matrix: (np.array) Covariance Matrix
    :param ranked_returns: (pd.DataFrame) ranked returns
    :param molecule: (list) Indices of quadruples
    :return: (pd.DataFrame) total euclidean distance for each quadruple
    """
    results = []
    for quadruple in molecule:
        results.append((quadruple, extremal_measure(ranked_returns[quadruple], co_variance_matrix)))

    return pd.DataFrame(results, columns=['quadruple', 'result'])


def run_extremal_measure_calcs(ranked_returns: pd.DataFrame, quadruples: list, co_variance_matrix: np.array, num_threads: int = 8,
                               verbose: bool = True) -> pd.DataFrame:
    """
    This function is the multi threading wrapper that supplies _extremal_calcs_loop with quadruples.
    :param co_variance_matrix: (np.array) Covariance Matrix
    :param ranked_returns: (pd.DataFrame) ranked returns
    :param quadruples: (list)  list of quadruples
    :param num_threads: (int) Number of cores to use
    :param verbose: (bool) Flag to report progress on asynch jobs
    :return: (pd.DataFrame) Quadruple with biggest extremal measure
    """

    chunk_size = len(quadruples) // num_threads
    quadruple_chunks = [quadruples[i:i + chunk_size] for i in range(0, len(quadruples), chunk_size)]

    with mp.Pool(num_threads) as p:
        results_list = p.starmap(_extremal_measure_loop, zip(repeat(ranked_returns), repeat(co_variance_matrix), quadruple_chunks))

    results = pd.concat(results_list)

    return results.iloc[results['result'].argmax()]
