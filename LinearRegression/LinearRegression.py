import numpy as np 
import math
class LinearRegression: 
    def __init__(self): 
        self.w = None
        self.b = None

    def cost_function(self, X, Y, w, b): 
        m = X.shape[0]
        cost = 0
        for i in range(m): 
            cost += ( np.dot(w, X) + b - Y[i] ) ** 2

        cost = cost/(2*m)
        return cost
    
    def gradient(self, X, Y, w, b): 
        m = X.shape[1]

        dj_dw = 0
        dj_db = 0

        for i in range(m): 

            f_wb = np.dot(w, X[i]) + b

            dj_dw_i = (f_wb - Y[i]) * X[i]
            dj_db_i = (f_wb - Y[i])
            dj_dw += dj_dw_i
            dj_db += dj_db_i
        
        dj_dw /= m
        dj_db /= m

        return (dj_dw, dj_db)
        

    def fit(self, X, Y, learning_rate = 0.001, iters = 10): 
        m = X.shape[1]

        w = np.zeros(m, dtype=float)
        b = 0

        for i in range(iters): 
            dj_dw, dj_db = self.gradient(X, Y, w, b)
            w = w - learning_rate * dj_dw
            b = b - learning_rate * dj_db

        self.w = w
        self.b = b

    def predict(self, X): 
        return np.dot(self.w, X) + self.b