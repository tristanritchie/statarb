# -*- coding: utf-8 -*-
"""
Generates corrleation matrix for specified ticker symbols

"""

import yfinance as yf
import pandas as pd
import numpy as np

ticker1_name = 'aapl'
ticker2_name = 'amzn'

data = yf.download(
        tickers = ticker1_name +' '+ ticker2_name,
        start = "2020-01-01",
        end = "2021-01-01",
        interval = "1d",
        group_by = 'ticker',
        auto_adjust = False,
        prepost = False,
        threads = True,
        proxy = None
    )

