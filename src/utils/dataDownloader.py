# coding=utf-8

"""
Goal: Downloading financial data (related to stock markets) from diverse sources
      (Alpha Vantage, Yahoo Finance).
Authors: Thibaut Théate and Damien Ernst
Institution: University of Liège
"""

import pandas as pd
import yfinance as yf  # Changed from pandas_datareader to yfinance
import requests
from io import StringIO

class AlphaVantage:
    """
    GOAL: Downloading stock market data from the Alpha Vantage API. See the
          AlphaVantage documentation for more information.
    """
    def __init__(self):
        self.link = 'https://www.alphavantage.co/query'
        self.apikey = 'your_api_key'
        self.datatype = 'csv'
        self.outputsize = 'full'
        self.data = pd.DataFrame()

    def getDailyData(self, marketSymbol, startingDate, endingDate):
        payload = {'function': 'TIME_SERIES_DAILY_ADJUSTED', 'symbol': marketSymbol,
                   'outputsize': self.outputsize, 'datatype': self.datatype,
                   'apikey': self.apikey}
        response = requests.get(self.link, params=payload)
        csvText = StringIO(response.text)
        data = pd.read_csv(csvText)
        
        # Handling missing or improperly named 'timestamp' column
        if 'timestamp' in data.columns:
            data.set_index('timestamp', inplace=True)
        else:
            # Infer datetime column if not properly named
            potential_date_columns = ['date', 'Date', 'datetime', 'Datetime']
            for col in potential_date_columns:
                if col in data.columns:
                    data.rename(columns={col: 'timestamp'}, inplace=True)
                    data.set_index('timestamp', inplace=True)
                    break
        
        self.data = self.processDataframe(data)
        if startingDate != 0 and endingDate != 0:
            self.data = self.data.loc[startingDate:endingDate]
        # Print the head of the data for verification
        # print(f"Head of the data for {marketSymbol}:")
        # print(self.data.head())
        return self.data

    def getHourlyData(self, marketSymbol, startingDate, endingDate):
        payload = {'function': 'TIME_SERIES_INTRADAY', 'symbol': marketSymbol,
                   'interval': '60min', 'outputsize': self.outputsize,
                   'datatype': self.datatype, 'apikey': self.apikey}
        response = requests.get(self.link, params=payload)
        csvText = StringIO(response.text)
        data = pd.read_csv(csvText)
        
        # Handling missing or improperly named 'timestamp' column
        if 'timestamp' in data.columns:
            data.set_index('timestamp', inplace=True)
        else:
            # Infer datetime column if not properly named
            potential_date_columns = ['date', 'Date', 'datetime', 'Datetime']
            for col in potential_date_columns:
                if col in data.columns:
                    data.rename(columns={col: 'timestamp'}, inplace=True)
                    data.set_index('timestamp', inplace=True)
                    break
        
        self.data = self.processDataframe(data)
        if startingDate != 0 and endingDate != 0:
            self.data = self.data.loc[startingDate:endingDate]
        # Print the head of the data for verification
        # print(f"Head of the data for {marketSymbol}:")
        # print(self.data.head())
        
        return self.data

    def processDataframe(self, dataframe):
        dataframe = dataframe[::-1]
        dataframe['close'] = dataframe['adjusted_close']
        del dataframe['adjusted_close']
        del dataframe['dividend_amount']
        del dataframe['split_coefficient']
        dataframe.index.names = ['Timestamp']
        dataframe = dataframe.rename(columns={"open": "Open", "high": "High",
                                              "low": "Low", "close": "Close",
                                              "volume": "Volume"})
        dataframe.index = dataframe.index.map(pd.Timestamp)
        return dataframe

class YahooFinance:
    def __init__(self):
        self.data = pd.DataFrame()

    def getDailyData(self, marketSymbol, startingDate, endingDate):
        # print(f"Fetching daily data from Yahoo Finance for {marketSymbol}...")
        data = yf.download(marketSymbol, start=startingDate, end=endingDate, interval='1d')
        if data.empty:
            raise ValueError("No daily data fetched from Yahoo Finance.")
        self.data = self.processDataframe(data)
        # Print the head of the data for verification
        # print(f"Head of the daily data for {marketSymbol}:")
        # print(self.data.head())
        return self.data

    def getHourlyData(self, marketSymbol, startingDate, endingDate):
        # print(f"Fetching hourly data from Yahoo Finance for {marketSymbol}...")
        data = yf.download(marketSymbol, start=startingDate, end=endingDate, interval='1h')
        if data.empty:
            raise ValueError("No hourly data fetched from Yahoo Finance.")
        self.data = self.processDataframe(data)
        # Print the head of the data for verification
        # print(f"Head of the hourly data for {marketSymbol}:")
        # print(self.data.head())
        return self.data

    def processDataframe(self, dataframe):
        dataframe['Close'] = dataframe['Adj Close']
        del dataframe['Adj Close']
        dataframe.index.names = ['Timestamp']
        dataframe = dataframe[['Open', 'High', 'Low', 'Close', 'Volume']]
        return dataframe

class CSVHandler:
    def dataframeToCSV(self, name, dataframe):
        path = name + '.csv'
        dataframe.to_csv(path)

    def CSVToDataframe(self, name):
        path = name + '.csv'
        return pd.read_csv(path, header=0, index_col='Timestamp', parse_dates=True)
