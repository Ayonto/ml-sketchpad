import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split

# import matplotlib.pyplot as plt

from LogisticRegression import LogisticRegression

data = pd.read_csv("LogisticRegression/accident.csv")
data = data.dropna()

data["Helmet_Used"] = data["Helmet_Used"].map({"Yes": 1, "No": 0})
data["Seatbelt_Used"] = data["Seatbelt_Used"].map({"Yes": 1, "No": 0})
data["Gender"] = data["Gender"].map({"Male": 1, "Female": 0})

Y = data["Survived"].to_numpy()
X = data.drop(["Survived"], axis=1).to_numpy()


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

# print(x_train)


model = LogisticRegression()

model.fit(x_train, y_train, learning_rate = 0.000001, iters = 1000)

result = model.predict(x_test)
print(f"score : {model.score(x_test, y_test)}")
model.show_learning_curve()
print(result)
print(y_test)



