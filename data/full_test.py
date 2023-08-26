import pandas as pd
import numpy as np

from partner_selection import PartnerSelection
from ps_utils import get_sector_data
from strategy.CMPI_strategy import CMPI


## Partner selection
df = pd.read_csv('scrap_book/data/data.csv', parse_dates=True, index_col='Date').dropna(how='any')
df_selection = df['2016-01-01':'2016-12-31'] #Taking 12 month data as mentioned in the paper
ps = PartnerSelection(df_selection)

constituents = pd.read_csv('scrap_book/data/constituents-detailed.csv', index_col='Symbol')

quadruples = ps.extremal(20)
print(quadruples)


## Initialise copula
q = 1

training_returns = df['2017-01-01':'2017-12-31'].loc[:, quadruples[q]].apply(lambda x: np.log(x).diff()).cumsum().dropna(how='any')

strat = CMPI()
strat.init_copula_model(training_returns)
print(strat.cvm.summary())

## Generate trading signals
testing_returns = df['2017-01-01':'2019-12-31'].loc[:, quadruples[q]].apply(lambda x: np.log(x).diff()).cumsum().dropna(how='any')

cmpi = strat.generate_cmpi(testing_returns)

