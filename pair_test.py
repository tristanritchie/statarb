# -*- coding: utf-8 -*-
"""
Correlation matrix, cointegration test for a ticker pair

"""

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ticker1_name = 'RIO.L'
ticker2_name = 'RIO.AX'

start_date = "2020-01-01"
end_date = "2021-01-01"

ticker1 = yf.Ticker(ticker1_name)
ticker2 = yf.Ticker(ticker2_name)

data1 = ticker1.history(start=start_date, end=end_date, frequency='1dy')['Close'].rename('ticker1_name')
data2 = ticker2.history(start=start_date, end=end_date, frequency='1dy')['Close'].rename('ticker2_name')

df = pd.concat([data1, data2], axis=1).bfill()

df.plot()








