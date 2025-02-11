from LinearRegression import LinearRegression
import pandas as pd
import numpy as np 


df = pd.read_csv('LinearRegression/dataset.csv')

d = df.experience

# Convert Age & Experience to a NumPy array
x_train = df[['age', 'experience']].to_numpy()  # More efficient than .values.tolist()

# Convert Salary to a NumPy array
y_train = df['income'].to_numpy()



# model fitting
model = LinearRegression()
model.fit(x_train, y_train)

print(model.predict([25,2]))