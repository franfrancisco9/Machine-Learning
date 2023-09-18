# imports
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, r2_score
from sklearn.linear_model import Lasso, Ridge, LinearRegression, ElasticNet, SquaredLoss, LogisticRegression, BayesianRidge
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# load training dataset
X_train = np.load("X_train_regression1.npy")
Y_train = np.load("y_train_regression1.npy")



# load test dataset
X_test_final = np.load("X_test_regression1.npy")

# Split
# X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.01, random_state=95789)

# Lasso
print("Lasso")
regressor = Lasso(
    random_state=95789,
    alpha = 0.001
)
regressor.fit(X_train, Y_train)
# Y_pred = regressor.predict(X_test)
Y_train_pred = regressor.predict(X_train)
SSE = 0
for i in range(len(Y_train)):
    SSE += (Y_train[i] - Y_train_pred[i])**2
print("Train:", SSE)
# print("Test:", r2_score(Y_test, Y_pred))

# Ridge
print("Ridge")
regressor = Ridge(
    random_state=95789,
    alpha = 0.1
)
regressor.fit(X_train, Y_train)
# Y_pred = regressor.predict(X_test)
Y_train_pred = regressor.predict(X_train)
SSE = 0
for i in range(len(Y_train)):
    SSE += (Y_train[i] - Y_train_pred[i])**2
print("Train:", SSE)
# print("Test:", r2_score(Y_test, Y_pred))

# LinearRegression
print("LinearRegression")
regressor = LinearRegression(

)
regressor.fit(X_train, Y_train)
# Y_pred = regressor.predict(X_test)
Y_train_pred = regressor.predict(X_train)
SSE = 0
for i in range(len(Y_train)):
    SSE += (Y_train[i] - Y_train_pred[i])**2
print("Train:", SSE)
# print("Test:", r2_score(Y_test, Y_pred))

# BayesianRidge
print("BayesianRidge")
regressor = BayesianRidge(
    alpha_1=0.0001,
    alpha_2=0.0001,
    lambda_1=0.0001,
    lambda_2=0.0001,
    tol=0.0001,
    compute_score=True,
    fit_intercept=True
)
regressor.fit(X_train, Y_train)
# Y_pred = regressor.predict(X_test)
Y_train_pred = regressor.predict(X_train)
SSE = 0
for i in range(len(Y_train)):
    SSE += (Y_train[i] - Y_train_pred[i])**2
print("Train:", SSE)

# ElasticNet
# regressor = ElasticNet(
#     random_state=95789,
#     alpha=0,
#     max_iter=100000
# )
# regressor.fit(X_train, Y_train)
# Y_pred = regressor.predict(X_test)
# Y_train_pred = regressor.predict(X_train)
# score = r2_score(Y_train, Y_train_pred)
# print("Alpha:", 0)
# print("ElasticNet:", score)