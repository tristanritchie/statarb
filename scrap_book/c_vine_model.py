# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 09:21:59 2018

@author: yili.peng
"""
import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr


class CVineModel:
    def __init__(self, lib_loc="~/R/x86_64-pc-linux-gnu-library/4.1/VineCopula"):
        """
        R and R library VineCopula are required!

        returns: all cadicate stock returns that contains target and partners during pair and formation period
        target: target ticker
        partners: 3 partner tickers in str format(like 'a,b,c') or in a list.
        lib_loc: location of R library VineCopula. Usually as ".../R-3.4.2/library"
        """
        pandas2ri.activate()
        self.vc = importr(name="VineCopula")
        self.cvm = None

    def fit(self, U, family=None):
        """
        training C-Vine Copula Modeling
        U: relative rank dataframe (n by p) entre between [0,1]. First column is the target stock.
        family: list or None(default). None means all possible familys.
        """

        family = ro.NA_Integer if family is None else ro.IntVector(family)

        quantiles_df = pandas2ri.DataFrame(U)
        dimension = U.shape[1]
        c_vine_structure_matrix = []
        for i in range(1, dimension + 1):
            c_vine_structure_matrix.extend([i] * i + [0] * (dimension - i))

        m = ro.r.matrix(
            ro.IntVector(c_vine_structure_matrix), nrow=dimension, byrow=True
        )
        cvm = self.vc.RVineCopSelect(quantiles_df, familyset=family, Matrix=m)
        self.cvm = cvm

    def predict(self, U, cvm=None):
        """
        compute h for a cvm given U
        h:  P( X1<x1 |x2,x3,...,xp)
        U: relative rank dataframe (n * p). First column is the target stock. U = (x1,x2,x3,...,xp)
        """
        cvm = self.cvm if cvm is None else cvm

        D = dict(cvm.items())
        dimension = len(D["names"])
        family = D["family"]
        par = D["par"]
        par2 = D["par2"]

        C_list = pd.DataFrame(
            None, columns=range(dimension - 1), index=range(dimension - 1)
        )
        for i in range(dimension, 1, -1):
            for j in range(i - 1, 0, -1):
                C_list.iloc[i - 2, j - 1] = self.vc.BiCop(
                    family[i - 1, j - 1], par[i - 1, j - 1], par2[i - 1, j - 1]
                )

        F_list = pd.DataFrame(None, columns=range(dimension), index=range(dimension))
        for i in range(dimension):
            F_list.iloc[dimension - 1, i] = ro.FloatVector(U.iloc[:, i].tolist())
        for i in range(dimension, 1, -1):
            for j in range(i - 1, 0, -1):
                F_list.iloc[i - 2, j - 1] = ro.FloatVector(
                    self.vc.BiCopHfunc1(
                        u1=F_list.iloc[i - 1, i - 1],
                        u2=F_list.iloc[i - 1, j - 1],
                        obj=C_list.iloc[i - 2, j - 1],
                    )
                )

        h = list(F_list.iloc[0, 0])
        return pd.Series(h, index=U.index, name=U.columns[0])

    def summary(self, cvm=None):
        cvm = self.cvm if cvm is None else cvm
        print(ro.r["summary"](cvm))

    def family(self, cvm=None):
        cvm = self.cvm if cvm is None else cvm
        D = dict(cvm.items())
        return np.array(D["family"])

    def par(self, cvm=None):
        cvm = self.cvm if cvm is None else cvm
        D = dict(cvm.items())
        return np.array(D["par"])

    def par2(self, cvm=None):
        cvm = self.cvm if cvm is None else cvm
        D = dict(cvm.items())
        return np.array(D["par2"])

    def aic(self, cvm=None):
        cvm = self.cvm if cvm is None else cvm
        D = dict(cvm.items())
        return list(D["AIC"])[0]

    def bic(self, cvm=None):
        cvm = self.cvm if cvm is None else cvm
        D = dict(cvm.items())
        return list(D["BIC"])[0]

    def names(self, cvm=None):
        cvm = self.cvm if cvm is None else cvm
        return list(cvm.names)

    def get_attr(self, name, cvm=None):
        cvm = self.cvm if cvm is None else cvm
        D = dict(cvm.items())
        return np.array(D[name])
