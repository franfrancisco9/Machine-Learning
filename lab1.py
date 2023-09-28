# imports
import numpy as np
# import k cross validation
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, r2_score
from sklearn.linear_model import Lasso, Ridge, LinearRegression, ElasticNet, SquaredLoss, LogisticRegression, BayesianRidge
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# # Leave One Take one out
# def leave_one_out(X, Y, i):
#     X_train = []
#     Y_train = []
#     X_test = []
#     Y_test = []
#     for j in range(len(X)):
#         if j == i:
#             X_test.append(X[j])
#             Y_test.append(Y[j])
#         else:
#             X_train.append(X[j])
#             Y_train.append(Y[j])
#     return np.array(X_train), np.array(X_test), np.array(Y_train), np.array(Y_test)

def press_statistic(y_true, y_pred, xs):
    res = y_pred - y_true
    hat = xs.dot(np.linalg.pinv(xs))
    den=(1-np.diagonal(hat))
    if np.any(den<=0):
        return np.inf
    sqr = np.square(res/den)
    return sqr.sum()

def predicted_r2(y_true, y_pred, xs):
    press = press_statistic(y_true, y_pred, xs)
    sst = np.square(y_true - y_true.mean()).sum()
    return 1- press/sst

def adj_r2_score(n, p, r2):
    return 1 - (1- r2)*(n-1)/(n-p-1)
# load training dataset
X_train = np.load("X_train_regression1.npy")
Y_train = np.load("y_train_regression1.npy")

# Visualize the different features
# for i in range(10):
#     X = []
#     for j in range(15):
#         X.append(X_train[j][i])
#     plt.figure()
#     plt.scatter(X, Y_train)
#     plt.savefig("./images/X" +str(i))

# load test dataset
X_test_final = np.load("X_test_regression1.npy")

# Split
# Seems to not make sense to split the data
# Since we have a small dataset, we will not split the data
# X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.01)

# # Lasso
# print("Lasso")
# best_SSE = np.inf
# best_R2_Pred = -np.inf
# for alp in np.linspace(0, 10, 10000):
#     regressor = Lasso(
#         random_state=95789,
#         alpha = alp
#     )
#     regressor.fit(X_train, Y_train)
#     # Y_pred = regressor.predict(X_test)
#     Y_train_pred = regressor.predict(X_train)
#     SSE = 0
#     for i in range(len(Y_train)):
#         SSE += (Y_train[i] - Y_train_pred[i])**2
#     r2_pred = predicted_r2(Y_train, Y_train_pred, X_train)
#     if SSE[0] < best_SSE:
#         print("Found better SSE")
#         print("Train:", SSE[0])
#         print("Alpha:", alp)
#         print("Test:", r2_score(Y_train, Y_train_pred))
#         print("Test adj:",adj_r2_score(15, 10, r2_score(Y_train, Y_train_pred)))
#         print("Test pred:",predicted_r2(Y_train, Y_train_pred, X_train))
#         best_SSE = SSE[0]
#     if r2_pred > best_R2_Pred:
#         print("Found better R2 pred")
#         print("Train:", SSE[0])
#         print("Alpha:", alp)
#         print("Test:", r2_score(Y_train, Y_train_pred))
#         print("Test adj:",adj_r2_score(15, 10, r2_score(Y_train, Y_train_pred)))
#         print("Test pred:",predicted_r2(Y_train, Y_train_pred, X_train))
#         best_R2_Pred = r2_pred
        
# # Ridge
# print("=========================Ridge===========================")
# best_SSE = np.inf
# best_R2_Pred = -np.inf
# for alp in np.linspace(0, 10, 10000):
#     regressor = Ridge(
#         random_state=95789,
#         alpha = alp
#     )
#     regressor.fit(X_train, Y_train)
#     # Y_pred = regressor.predict(X_test)
#     Y_train_pred = regressor.predict(X_train)
#     SSE = 0
#     for i in range(len(Y_train)):
#         SSE += (Y_train[i] - Y_train_pred[i])**2
#     r2_pred = predicted_r2(Y_train, Y_train_pred, X_train)
#     if SSE[0] < best_SSE:
#         print("Found better SSE")
#         print("Train:", SSE[0])
#         print("Alpha:", alp)
#         print("Test:", r2_score(Y_train, Y_train_pred))
#         print("Test adj:",adj_r2_score(15, 10, r2_score(Y_train, Y_train_pred)))
#         print("Test pred:",predicted_r2(Y_train, Y_train_pred, X_train))
#         best_SSE = SSE[0]
#     if r2_pred > best_R2_Pred:
#         print("Found better R2 pred")
#         print("Train:", SSE[0])
#         print("Alpha:", alp)
#         print("Test:", r2_score(Y_train, Y_train_pred))
#         print("Test adj:",adj_r2_score(15, 10, r2_score(Y_train, Y_train_pred)))
#         print("Test pred:",predicted_r2(Y_train, Y_train_pred, X_train))
#         best_R2_Pred = r2_pred

# Lasso
print("Lasso")
regressor = Lasso(
    random_state=95789,
    alpha = 0.05
)
regressor.fit(X_train, Y_train)
# Y_pred = regressor.predict(X_test)
Y_train_pred = regressor.predict(X_train)
SSE = 0
for i in range(len(Y_train)):
    SSE += (Y_train[i] - Y_train_pred[i])**2
print("Train:", SSE)
print("Test:", r2_score(Y_train, Y_train_pred))
print("Test adj:",adj_r2_score(15, 10, r2_score(Y_train, Y_train_pred)))
print("Test pred:",predicted_r2(Y_train, Y_train_pred, X_train))
print("Coefficients:", regressor.coef_)

# Ridge
print("Ridge")
regressor = Ridge(
    random_state=95789,
    alpha = 0.05
)
regressor.fit(X_train, Y_train)
# Y_pred = regressor.predict(X_test)
Y_train_pred = regressor.predict(X_train)
SSE = 0
for i in range(len(Y_train)):
    SSE += (Y_train[i] - Y_train_pred[i])**2
print("Train:", SSE)
print("Test:", r2_score(Y_train, Y_train_pred))
print("Test adj:",adj_r2_score(15, 10, r2_score(Y_train, Y_train_pred)))
print("Test pred:",predicted_r2(Y_train, Y_train_pred, X_train))
print("Coefficients:", regressor.coef_)
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
print("Test:", r2_score(Y_train, Y_train_pred))
print("Test adj:",adj_r2_score(15, 10, r2_score(Y_train, Y_train_pred)))
print("Test pred:",predicted_r2(Y_train, Y_train_pred, X_train))
print("Coefficients:", regressor.coef_)

# do K-fold cross validation with linear regression
print("======================LinearRegression======================")
best_SSE = np.inf
best_R2_Pred = -np.inf
for k in range(3, 6):
    kf = KFold(n_splits=k, shuffle=True)
    for train_index, test_index in kf.split(X_train):
        # print("K:", k)
        # print(len(list(kf.split(X_train))))
        # print(list(kf.split(X_train)))
        X_train_cv, X_test_cv = X_train[train_index], X_train[test_index]
        Y_train_cv, Y_test_cv = Y_train[train_index], Y_train[test_index]
        regressor = LinearRegression()
        regressor.fit(X_train_cv, Y_train_cv)
        Y_pred_test = regressor.predict(X_test_cv)
        Y_pred = regressor.predict(X_train_cv)
        SSE = 0
        for i in range(len(Y_test_cv)):
            SSE += (Y_test_cv[i] - Y_pred_test[i])**2
        if SSE[0] < best_SSE :
            print("Found better SSE")
            print("K:", k)
            print("train_index:", train_index)
            print("test_index:", test_index)
            print("SSE:", SSE[0])
            print("R2:", r2_score(Y_train_cv, Y_pred))
            print("R2 adj:",adj_r2_score(15, 10, r2_score(Y_train_cv, Y_pred)))
            print("R2 pred:",predicted_r2(Y_train_cv, Y_pred, X_train_cv))
            best_SSE = SSE[0]

# do k-fold cross validation with lasso
print("======================Lasso======================")
best_SSE = np.inf
best_R2_Pred = -np.inf
for k in range(3, 6):
    kf = KFold(n_splits=k, shuffle=True)
    for train_index, test_index in kf.split(X_train):
        # print("K:", k)
        # print(len(list(kf.split(X_train))))
        # print(list(kf.split(X_train)))
        X_train_cv, X_test_cv = X_train[train_index], X_train[test_index]
        Y_train_cv, Y_test_cv = Y_train[train_index], Y_train[test_index]
        regressor = Lasso(
            random_state=95789,
            alpha = 0.05
        )
        regressor.fit(X_train_cv, Y_train_cv)
        Y_pred_test = regressor.predict(X_test_cv)
        Y_pred = regressor.predict(X_train_cv)
        SSE = 0
        for i in range(len(Y_test_cv)):
            SSE += (Y_test_cv[i] - Y_pred_test[i])**2
        if SSE[0] < best_SSE :
            print("Found better SSE")
            print("K:", k)
            print("train_index:", train_index)
            print("test_index:", test_index)
            print("SSE:", SSE[0])
            print("R2:", r2_score(Y_train_cv, Y_pred))
            print("R2 adj:",adj_r2_score(15, 10, r2_score(Y_train_cv, Y_pred)))
            print("R2 pred:",predicted_r2(Y_train_cv, Y_pred, X_train_cv))
            best_SSE = SSE[0]

# do k-fold cross validation with ridge
print("======================Ridge======================")
best_SSE = np.inf
best_R2_Pred = -np.inf
for _ in range(1000):
    for k in range(3, 6):
        kf = KFold(n_splits=k, shuffle=True)
        for train_index, test_index in kf.split(X_train):
            # test alpha from 0.01 up to 10 giong 0.01
            for alpha in np.linspace(0.01, 10, 1000):
                # print("K:", k)
                # print(len(list(kf.split(X_train))))
                # print(list(kf.split(X_train)))
                X_train_cv, X_test_cv = X_train[train_index], X_train[test_index]
                Y_train_cv, Y_test_cv = Y_train[train_index], Y_train[test_index]
                regressor = Ridge(
                    random_state=95789,
                    alpha = alpha
                )
                regressor.fit(X_train_cv, Y_train_cv)
                Y_pred_test = regressor.predict(X_test_cv)
                Y_pred = regressor.predict(X_train_cv)
                SSE = 0
                for i in range(len(Y_test_cv)):
                    SSE += (Y_test_cv[i] - Y_pred_test[i])**2
                if SSE[0] < best_SSE and predicted_r2(Y_train_cv, Y_pred, X_train_cv) > best_R2_Pred:
                    print("Found better SSE")
                    print("K:", k)
                    print("train_index:", train_index)
                    print("test_index:", test_index)
                    print("Alpha:", alpha)
                    print("SSE:", SSE[0])
                    print("R2:", r2_score(Y_train_cv, Y_pred))
                    print("R2 adj:",adj_r2_score(15, 10, r2_score(Y_train_cv, Y_pred)))
                    print("R2 pred:",predicted_r2(Y_train_cv, Y_pred, X_train_cv))
                    best_SSE = SSE[0]
                    best_R2_Pred = predicted_r2(Y_train_cv, Y_pred, X_train_cv)


