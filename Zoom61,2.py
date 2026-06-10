import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score

from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

dataset = pd.read_csv("Zooms\DeepLearning\studentgrades.csv")

l_e = LabelEncoder()
l_e_cols = ["name","grade"]
for col in l_e_cols:
    dataset[col] = l_e.fit_transform(dataset[col])

x = dataset.drop("grade",axis=1)
y = dataset["grade"]

x_tr,x_ts,y_tr,y_ts = train_test_split(x,y,test_size=0.2,random_state=5)

regressor = RandomForestRegressor(n_estimators=500,random_state=0)
regressor.fit(x_tr,y_tr)

y_pred = regressor.predict(x_ts)

mse = mean_squared_error(y_ts, y_pred)
r2 = r2_score(y_ts,y_pred)

print("Mean Square Error:",mse)
print("R-Squared:",r2)

plt.figure(figsize=(10,5))
plt.scatter(range(len(y_ts)),    y_ts,color="blue",label="Actual")
plt.scatter(range(len(y_pred)),y_pred,color="red", label="Predicted")
plt.title("Actual vs Predicted")
plt.xlabel("Student Index")
plt.ylabel("Final Grade (G3)")
plt.legend()
plt.show()

encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y.values.reshape(-1,1))
print(y_encoded)

x_tr_nn, x_ts_nn, y_tr_nn, y_ts_nn = train_test_split(x,y_encoded,test_size=0.2,random_state=42)

nn_model = Sequential([
    Dense(16, activation="relu", input_shape=(4,)),
    Dense(16, activation="relu"),
    Dense(4,  activation="softmax")
])

nn_model.compile(optimizer="adam",
                 loss="categorical_crossentropy",
                 metrics=["accuracy"])

nn_model.fit(x_tr_nn,y_tr_nn,epochs=50,batch_size=5,validation_split=0.1,verbose=1)

nn_lose, nn_accuracy = nn_model.evaluate(x_ts_nn, y_ts_nn, verbose=1)

print("Neural Network Accuracy (DL):", nn_accuracy)