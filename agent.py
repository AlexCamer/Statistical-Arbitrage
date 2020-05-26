import itertools
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
from sklearn.linear_model import LinearRegression
from kalman_filter import *





# cointegrates two series, returns intercept and slope
def cointegrate(series_1, series_2):
    regressor = LinearRegression()
    regressor.fit(series_2.values.reshape((-1,1)), series_1.values)
    return (regressor.intercept_, regressor.coef_[0])


# models Ornstein-Uhlenbeck process as an Autoregressive(1) series
#   to find mean-reversion spead
def get_reversion_speed(series):
    df = pd.DataFrame(series, columns=['y'])
    df['x'] = df['y'].shift(1)
    df.dropna(inplace=True)
    regressor = LinearRegression()
    regressor.fit(df['x'].values.reshape((-1,1)), df['y'].values)
    b = regressor.coef_[0]
    k = - np.log(b) if b > 0 else 0
    return k





# Statistical Arbitrage Trading Agent
class Agent:

    pairs = {}
    sectors = None
    trade_size = None


    #
    def __init__(self, sectors, trade_size):
        self.sectors = sectors
        self.trade_size = trade_size


    # returns possible pairs to trade on, identified by sector 
    #   and correlation
    def get_possible_pairs(self, prices):
        symbols = set(prices.columns)
        returns = (prices - prices.shift(1)).dropna()
        possible_pairs = []
        for sector in self.sectors.values():
            valid_symbols = list(symbols.intersection(sector))
            if len(valid_symbols) < 2: continue
            valid_symbols.sort()

            # get correlation matrix of sector
            sector_returns = returns[valid_symbols]
            corr = sector_returns.corr()

            # mask repeated values in correlation matrix
            mask = np.triu(np.ones(corr.shape), 1).astype(bool)
            corr = corr.where(mask)

            # flatten correlation matrix into vector
            stacked = corr.stack().reset_index()
            stacked.columns = ['symbol_1', 'symbol_2', 'corr']

            # filter out pairs by correlation
            correlated = stacked.loc[stacked['corr'] > 0.7]
            possible_pairs += zip(correlated['symbol_1'], correlated['symbol_2'])
        return possible_pairs

    
    # identifies valid pairs and updates self.pairs
    def calibrate(self, prices):

        # identify valid pairs
        possible_pairs = self.get_possible_pairs(prices)
        valid, betas = [], []
        for pair in possible_pairs:
            
            # Engle-Granger Test
            log_1, log_2 = np.log(prices[list(pair)])._series.values()
            if ts.coint(log_1, log_2)[1] > 0.05: continue

            # Augmented Dickey-Fuller Test
            alpha, beta = cointegrate(log_1, log_2)
            spread = log_1 - beta * log_2 - alpha
            if ts.adfuller(spread)[1] > 0.05: continue

            # Mean Reversion Test
            k = get_reversion_speed(spread)
            if k < 0.2: continue
            valid.append(pair)
        
        # remove invalid live pairs
        invalid = set(self.pairs.keys()) - set(valid)
        orders = []
        for pair in invalid:
            positions = np.array(self.pairs[pair]['positions'])
            if np.absolute(positions).sum() > 0: orders += list(zip(pair, positions * -1))
            del self.pairs[pair]

        # add new valid pairs
        for pair in valid:
            if pair not in self.pairs:
                training_data = np.log(prices[list(pair)].values[:-1])
                self.pairs[pair] = {
                    'positions': [0, 0],
                    'state': 'neutral',
                    'filter': Kalman_Filter(training_data)
                }
        return orders
    

    # gets integer quantity of securities to invest for the arbitrage
    def get_quantities(self, spot_prices, beta):
        weights = np.array([beta, 1]) / (beta + 1)
        quantities = weights * self.trade_size / spot_prices
        return quantities.astype(int)


    # return the amount currently being leveraged
    def get_leverage(self):
        lev = 0
        for pair in self.pairs.values():
            if pair['state'] != 'neutral': lev += self.trade_size / 2
        return lev

    
    # given new data, makes trading descisions and returns respective orders
    def get_orders(self, prices):
        orders, stops = [], []
        for pair, data in self.pairs.items():
            spot_prices = prices[list(pair)].values[-1]
            spread, std, beta = data['filter'].filter(np.log(spot_prices))
            order = np.zeros(2, dtype=int)

            # exit positions if spread falls under standard deviation
            if (data['state'] == 'long' and spread >= -std) or \
            (data['state'] == 'short' and spread <= std): 
                quantities = np.array(data['positions']) * -1
                self.pairs[pair]['positions'] = [0, 0]
                self.pairs[pair]['state'] = 'neutral'
                order += quantities
            
            # enter positions if spread exceeds standard deviation
            if data['state'] == 'neutral':
                if spread <= -std:
                    quantities = self.get_quantities(spot_prices, beta) * [1, -1]
                    self.pairs[pair]['positions'] = quantities.tolist()
                    self.pairs[pair]['state'] = 'long'
                    order += quantities
                elif spread >= std:
                    quantities = self.get_quantities(spot_prices, beta) * [-1, 1]
                    self.pairs[pair]['positions'] = quantities.tolist()
                    self.pairs[pair]['state'] = 'short'
                    order += quantities

            if np.abs(order).sum() > 0: orders += list(zip(pair, order))
        return orders, self.get_leverage()