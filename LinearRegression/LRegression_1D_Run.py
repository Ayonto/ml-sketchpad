from LinearRegression_1D import LinearRegression

import pandas as pd
import numpy as np 

# import matplotlib.pyplot as plt

data = pd.read_csv("LinearRegression/xyData.csv")

# Drop the missing values
data = data.dropna()


# training dataset and labels
train_input = np.array(data.x[0:500]).reshape(500)
train_output = np.array(data.y[0:500]).reshape(500)



# valid dataset and labels
test_input = np.array(data.x[500:700]).reshape(199)
test_output = np.array(data.y[500:700]).reshape(199)



# #model fitting
model = LinearRegression()
model.fit(train_input, train_output, 0.0001, 10000)

pred = model.predict(test_input)
len_test = len(test_input)
mse = sum((test_output[i] - pred[i])**2 for i in range(len_test)) / len_test

print(f"Mean Squared Error: {mse}")
print(f"New Feature Test x = 1345; y = {model.predict(1345)}")
