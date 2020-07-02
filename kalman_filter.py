import numpy as np
from sklearn.linear_model import LinearRegression



# Kalman filter class derived from (Ernest Chan, 2013)
class Kalman_Filter():

    # initilizes filter params and trains filter with obs data
    def __init__(self, obs):
        self.Vw = 3e-6 * np.eye(2)
        self.Ve = 1e-3
        self.beta = np.zeros(2)
        self.P = np.zeros((2,2))
        self.R = None
        for ob in obs: self.filter(ob)


    # calculates obs error, error deviation and hedge ratio
    def filter(self, ob):
        x, y = ob
        x = np.array([x,1]).reshape((1,2))
        if self.R is not None: self.R = self.P + self.Vw
        else: self.R = np.zeros((2,2))
        y_hat = (x @ self.beta).item()
        var = (x @ self.R @ x.T).item()
        std = np.sqrt(var)
        Q = var + self.Ve
        e = y - y_hat
        K = self.R @ x.T / Q
        self.beta += K.flatten() * e
        self.P = self.R - K @ x @ self.R
        return (-e, std, self.beta[0])
