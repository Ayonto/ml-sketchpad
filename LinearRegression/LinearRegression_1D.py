import numpy as np 
import math
class LinearRegression: 
    def __init__(self): 
        self.parameters = {"w": 0, "b": 0}
        print("1-Dimensional Linear Regression")


    def cost_function(self, X, Y, w, b): 
        m = len(X)
        cost = 0
        for i in range(m): 
            cost += (w * X[i] + b - Y[i] ) ** 2

        cost = cost/(2*m)
        return cost
    
    def gradient(self, X, Y, w, b): 
        m = len(X)

        dj_dw = 0
        dj_db = 0
        for i in range(m): 
            # print(f"w: {w}, b: {b}, X[i]: {X[i]}, dj_dw: {dj_dw}")

            f_wb = w * X[i] + b
            dj_dw_i = (f_wb - Y[i]) * X[i]
            dj_db_i = (f_wb - Y[i])
            dj_db += dj_db_i
            dj_dw += dj_dw_i 

        dj_dw = dj_dw / m
        dj_db = dj_db / m

        return (dj_dw, dj_db)
            
    def fit(self, X, Y, learning_rate, iters): 
        J_history = []
        p_history = []
        
        w = 0
        b = 0
        for i in range(iters): 
            dj_dw, dj_db = self.gradient(X, Y, w, b)
            
            w = w - learning_rate * dj_dw 
            b = b - learning_rate * dj_db
            w = np.clip(w, -1e10, 1e10)
            b = np.clip(b, -1e10, 1e10)



            if i<100000:      # prevent resource exhaustion 
                J_history.append( self.cost_function(X, Y, w , b))
                p_history.append([w,b])
            if i% math.ceil(iters/10) == 0:
                print(f"Iteration {i:4}: Cost {J_history[-1]} ",
                    f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                    f"w: {w: 0.3e}, b:{b: 0.5e}")
                    
                    
        self.parameters["w"] = w; self.parameters["b"] = b
        print(f"Parameters: {self.parameters}")

    def predict(self, x): 
        return self.parameters["w"] * x + self.parameters["b"]