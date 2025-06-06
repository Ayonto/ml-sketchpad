import numpy as np 
import matplotlib.pyplot as plt


class LogisticRegression: 
    def __init__(self, w = None, b = None): 
        self.w = w
        self.b = b

        self.J = []
        self.iters = []
    
    def sigmoid(self, x, w, b): 
        return 1 / (1 + np.exp(-(np.dot(x, w) + b)))

    def cost_function(self, X, Y, w, b): 
        m = X.shape[0]
        cost = 0
        for i in range(m): 
            z_i = np.dot(X[i], w) + b

            f_wb_i = self.sigmoid(X[i], w, b)
            cost += -Y[i] * np.log(f_wb_i) - (1 - Y[i]) * np.log(1-f_wb_i)

        cost = cost / m
        return cost
    
    def gradient(self, X, Y, w, b): 
        m, n = X.shape
        dj_dw = np.zeros(n)
        dj_db = 0
        # print(f"Debug: m {m}")
        for i in range(m): 
            f_wb = self.sigmoid(X[i], w, b)
            dj_dw = (f_wb - Y[i] ) * X[i]
            dj_db = f_wb - Y[i]
            dj_dw += dj_dw
            dj_db += dj_db

        dj_dw /= m
        dj_db /= m
        return dj_dw, dj_db

    def fit(self, X, Y, learning_rate, iters=1000): 
        m, n = X.shape
        w = np.zeros((n,))
        b = 0

        for i in range(iters): 
            dj_dw, dj_db = self.gradient(X, Y, w, b)


            w = w - learning_rate * dj_dw
            b = b - learning_rate * dj_db

            if i % 100 == 0:
                print(f"Iteration {i}: w = {w}, b = {b}, Cost: {self.cost_function(X, Y, w, b)}")
            if i% 10 == 0:
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

    def score(self, X_test, Y_test):
            # Predict using the trained model
            Y_pred = self.predict(X_test)

            # Compute R² Score
            ss_total = np.sum((Y_test - np.mean(Y_test)) ** 2)  # Total Sum of Squares
            ss_residual = np.sum((Y_test - Y_pred) ** 2)  # Residual Sum of Squares

            r2_score = 1 - (ss_residual / ss_total)

            return r2_score
    def predict(self, X): 
        print(f"shape of X: {X.shape}")
        return self.sigmoid(X, self.w, self.b)

            

    
