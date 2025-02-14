import numpy as np 
import math
from matplotlib import pyplot as plt
class LinearRegression: 
    def __init__(self): 
        self.w = None
        self.b = None
        
        self.J = []
        self.iters = []
        

    def cost_function(self, X, Y, w, b): 
        m = X.shape[0]
        cost = 0
        for i in range(m): 
            cost += ( np.dot(w, X[i]) + b - Y[i] ) ** 2

        cost = cost/(2*m)
        return cost
    
    def gradient(self, X, Y, w, b): 
        m = X.shape[0]
        n = X.shape[1]
        dj_dw = np.zeros(n)
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

        # for gradient descent Learning Curve
        self.J = []
        self.iters = []

        
        m = X.shape[0]
        n = X.shape[1]
        w = np.zeros(n, dtype=float)
        b = 0

        for i in range(iters): 
            dj_dw, dj_db = self.gradient(X, Y, w, b)
            # if i < 1000: 
            #     print(f"dj_dw {dj_dw}")
            w = w - learning_rate * dj_dw
            b = b - learning_rate * dj_db
            # w = np.clip(w, -1e10, 1e10)
            # b = np.clip(b, -1e10, 1e10)

            if i % 10 == 0:
                print(f"Iteration {i}: w = {w}, b = {b}, Cost: {self.cost_function(X, Y, w, b)}")

            if i% math.ceil(iters/10) == 0:
                self.J.append(self.cost_function(X, Y, w, b))

                self.iters.append(i)

        self.w = w
        self.b = b

    def show_learning_curve(self): 
        plt.plot(self.iters, self.J, marker = 'o', linestyle='-')
        plt.xlabel("Iterations")
        plt.ylabel("Cost Function J(w, b)")
        plt.title("Learning Curve")
        plt.grid(True)
        plt.legend()
        plt.show()

    def predict(self, X): 
        return np.dot(self.w, X) + self.b