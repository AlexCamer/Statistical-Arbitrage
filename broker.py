import numpy as np
import pandas as pd



# class for testing trading models 
class Broker:

    df = None
    window = None
    portfolio = None
    cash = None
    i = None


    # initialize starting values, portfolio weight are stored in a series
    def __init__(self, df, window, starting_cash=0):
        self.df = df
        self.window = window
        self.cash = starting_cash
        self.i = window - 1
        weights = np.zeros(len(df.columns), dtype=int)
        self.portfolio = pd.Series(weights, df.columns)


    # returns initial observation
    def init(self):
        self.i = self.window
        return self.df.iloc[0:self.i]


    # takes in orders, updates porfolio, returns next observations 
    def step(self, orders=[]):
        prices = self.df.iloc[self.i]
        for symbol, quantity in orders:
            price = quantity * prices[symbol]
            self.cash -= price
            self.portfolio[symbol] += quantity
        net_gain = self.get_net_gain()
        self.i += 1
        if self.i >= len(self.df): return None, net_gain, True
        return self.df.iloc[self.i-self.window:self.i], net_gain, False

    
    # returns value of portfolio + cash
    def get_net_gain(self):
        prices = None
        if self.i < len(self.df): prices = self.df.iloc[self.i]
        else: prices = self.df[-1]
        portfolio_value = np.sum(self.portfolio * prices)
        return portfolio_value + self.cash
