from LinearRegression import LinearRegression

import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as LR

df = pd.read_csv('LinearRegression/Student_Performance.csv')

# Converting "Yes" to 1 and "No" to 0
df["Extracurricular Activities"] = df["Extracurricular Activities"].replace({"Yes": 1, "No": 0})
# Ensure the column is of integer type
df["Extracurricular Activities"] = df["Extracurricular Activities"].astype(int)

# Convert to a NumPy array
X = df[['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced']].to_numpy()  

# Convert Output to a NumPy array
Y = df['Performance Index'].to_numpy()

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)



# model fitting
model = LinearRegression()
model.fit(x_train, y_train, learning_rate=0.00003, iters = 10001)

print("Model Score: ", model.score(x_test, y_test))

model.show_learning_curve()