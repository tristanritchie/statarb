import pandas as pd
import yfinance as yf
import os

from partner_selection import PartnerSelection
from ps_utils import get_sector_data


def Fetch_Returns(ticker_list, start_date, end_date):
    df = yf.download(' '.join(ticker_list), 
                        start=start_date, 
                        end=end_date, 
                        frequency='1dy',
                        auto_adjust = True,)['Close'].ffill()

    path = os.getcwd()+'\\data'
    os.makedirs(path, exist_ok=True)
    df.to_csv(path+'\\price.csv')


def Fetch_Info(ticker_list):
    info_list = []
    for stock in ticker_list:
        info = yf.Ticker(stock).info
        name = info.get('longName')
        sector = info.get('sector')
        country = info.get('country')
        cap = info.get('marketCap')
        info_list.append([stock, name, sector, country, cap])

    info_df = pd.DataFrame(info_list, columns = ['Symbol','Name','Sector','Country','Market Cap'])
    return info_df


def Partner_Selection(path):
    df = pd.read_csv(path, parse_dates=True, index_col='Date').dropna()
    ps = PartnerSelection(df)
    Q = ps.extremal(20)
    return Q