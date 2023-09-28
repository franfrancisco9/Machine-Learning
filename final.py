# Import libraries
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

def Linear_ordinary_regression(x_train, y_train, x_test):
    """
    Function that performs ordinary linear regression.
    Input:
        x_train: matrix of features for training
        y_train: vector of outcomes for training
        x_test: matrix of features for testing
    Output as a tuple:
        y_pred: vector of predicted outcomes for the given train features
        SSE: sum of squared errors for the given train features
        y_test: vector of predicted outcomes for the given test features
    """
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)

    # Predict for the train features.
    y_pred = linear.predict(x_train)

    # Calculate SSE
    SSE = 0
    for i in range(y_train.shape[0]):
        SSE += (y_train[i] - y_pred[i])**2

    # Predict for the test features.
    y_test = linear.predict(x_test)
    
    return (y_pred, SSE, y_test)


# order - order of the polynomial
# x_train - feature vector
# y_train - vector of outcomes
# x_test - feature vector for testing and evaluating the used model
def polynomial_regression(order, x_train, y_train, x_test):
    """
    Function that performs polynomial regression.
    Input:
        order: order of the polynomial
        x_train: matrix of features for training
        y_train: vector of outcomes for training
        x_test: matrix of features for testing
    Output as a tuple:
        y_pred: vector of predicted outcomes for the given train features
        SSE: sum of squared errors for the given train features
        y_test: vector of predicted outcomes for the given test features
    """

    # Adjust the features to the polynomial order.
    X_train = x_train.copy() 
    poly = PolynomialFeatures(degree=order)
    X_train = poly.fit_transform(X_train)
    reg = linear_model.LinearRegression()
    reg.fit(X_train, y_train)

    # Predict for the train features.
    y_pred = reg.predict(X_train)

    SSE = 0
    # Calculate SSE
    for i in range(y_train.shape[0]):
        SSE += (y_train[i] - y_pred[i])**2

    X_test = x_test.copy() 
    poly = PolynomialFeatures(degree=order)
    X_test = poly.fit_transform(X_test)

    # Predict for the test features.
    y_test = reg.predict(X_test)

    return (y_pred, SSE, y_test)

def ridge(alpha, x_train, y_train, x_test):
    """
    Function that performs ridge regression.
    Input:
        alpha: regularization term
        x_train: matrix of features for training
        y_train: vector of outcomes for training
        x_test: matrix of features for testing
    Output as a tuple:
        y_pred: vector of predicted outcomes for the given train features
        SSE: sum of squared errors for the given train features
        y_test: vector of predicted outcomes for the given test features
    """
    ridge = linear_model.Ridge(alpha=alpha) 
    ridge.fit(x_train, y_train)

    # Predict for the train features.
    y_pred = ridge.predict(x_train)

    SSE = 0
    # Calculate SSE
    for i in range(y_train.shape[0]):
        SSE += (y_train[i] - y_pred[i])**2

    # Predict for the test features.
    y_test = ridge.predict(x_test)

    return (y_pred, SSE, y_test)

def lasso(alpha, x_train, y_train, x_test):
    """
    Function that performs lasso regression.
    Input:
        alpha: regularization term
        x_train: matrix of features for training
        y_train: vector of outcomes for training
        x_test: matrix of features for testing
    Output as a tuple:
        y_pred: vector of predicted outcomes for the given train features
        SSE: sum of squared errors for the given train features
        y_test: vector of predicted outcomes for the given test features
    """
    lasso = linear_model.Lasso(alpha=alpha) # The larger the value of alpha, the greater the amount of shrinkage of the coefficients
    lasso.fit(x_train, y_train)
 
    # Predict for the train features.
    y_pred = lasso.predict(x_train)
    
    # Calculate SSE
    SSE = 0
    for i in range(y_train.shape[0]):
        SSE += (y_train[i] - y_pred[i])**2

    # Predict for the test features.
    y_test = lasso.predict(x_test)

    return (y_pred, SSE, y_test)

def Kfolds_linear_ordinary_regression(Kfolds, x_train, y_train, nstates):
    """
    Function that performs ordinary linear regression with cross-validation.
    Input:
        Kfolds: number of folds
        x_train: matrix of features for training
        y_train: vector of outcomes for training
        nstates: number of times to shuffle data
    Output:
        SSE_avg_final: average sum of squared errors for the given train features
    """
    SSE_validation_ord = 0
    SSE_avg_final = 0

    for k in range(2, nstates+2):
        kf = KFold(n_splits=Kfolds, random_state=k, shuffle=True)
        for train_new_index, test_validation_index in kf.split(x_train):

            x_train_new, x_validation = x_train[train_new_index], x_train[test_validation_index]
            y_train_new, y_validation = y_train[train_new_index], y_train[test_validation_index]

            linear_regression = Linear_ordinary_regression(x_train_new, y_train_new, x_validation)

            # Calculate SSE with the validation set
            y_ord_test = linear_regression[2]
            for i in range(y_ord_test.shape[0]):
                SSE_validation_ord += (y_validation[i] - y_ord_test[i])**2
        
        SSE_validation_ord = SSE_validation_ord/(Kfolds)
        SSE_avg_final += SSE_validation_ord
        SSE_validation_ord = 0

    SSE_avg_final = SSE_avg_final/nstates

    return SSE_avg_final

def Kfolds_ridge_regression(Kfolds, x_train, y_train, alpha, nstates):
    """
    Function that performs ridge regression with cross-validation.
    Input:
        Kfolds: number of folds
        x_train: matrix of features for training
        y_train: vector of outcomes for training
        alpha: regularization term
        nstates: number of times to shuffle data
    Output:
        SSE_avg_final: average sum of squared errors for the given train features
    """
    SSE_validation_ridge = 0
    SSE_avg_final = 0

    for k in range(nstates):
        kf = KFold(n_splits=Kfolds, shuffle=True, random_state = k)
        for train_new_index, test_validation_index in kf.split(x_train):

            x_train_new, x_validation = x_train[train_new_index], x_train[test_validation_index]
            y_train_new, y_validation = y_train[train_new_index], y_train[test_validation_index]

            ridge_regression = ridge(alpha, x_train_new, y_train_new, x_validation)

            # Calculate SSE with the validation set
            y_ridge_test = ridge_regression[2]
            for i in range(y_validation.shape[0]):
                SSE_validation_ridge += (y_validation[i] - y_ridge_test[i])**2

        SSE_validation_ridge = SSE_validation_ridge/(Kfolds)
        SSE_avg_final += SSE_validation_ridge
        SSE_validation_ridge = 0

    SSE_avg_final = SSE_avg_final/nstates

    return SSE_avg_final

def Kfolds_lasso_regression(Kfolds, x_train, y_train, alpha, nstates):
    """
    Function that performs lasso regression with cross-validation.
    Input:
        Kfolds: number of folds
        x_train: matrix of features for training
        y_train: vector of outcomes for training
        alpha: regularization term
        nstates: number of times to shuffle data
    Output:
        SSE_avg_final: average sum of squared errors for the given train features
    """
    SSE_validation_lasso = 0
    SSE_avg_final = 0

    for k in range(nstates):
        kf = KFold(n_splits=Kfolds, shuffle=True, random_state = k)

        for train_new_index, test_validation_index in kf.split(x_train):

            x_train_new, x_validation = x_train[train_new_index], x_train[test_validation_index]
            y_train_new, y_validation = y_train[train_new_index], y_train[test_validation_index]

            lasso_regression = lasso(alpha, x_train_new, y_train_new, x_validation)

            # Calculate SSE with the validation set
            y_lasso_test = lasso_regression[2]
            for i in range(y_validation.shape[0]):
                SSE_validation_lasso += (y_validation[i] - y_lasso_test[i])**2

        SSE_validation_lasso = SSE_validation_lasso/(Kfolds)
        SSE_avg_final += SSE_validation_lasso
        SSE_validation_lasso = 0

    SSE_avg_final = SSE_avg_final/nstates

    return(SSE_avg_final)

x_train = np.load('X_train_regression1.npy')
y_train = np.load('Y_train_regression1.npy')
x_test = np.load('X_test_regression1.npy')

order = 3 # order of the polynomial
lambda_ridge = 0.17600000000000002 # Regularization term for Ridge regression
lambda_lasso = 0.001 # Regularization term for Lasso regression

print("\nData shape of vector of features data and vector of outcomes data for training: ", x_train.shape, y_train.shape)
print("\nData shape of vector of features for prediction: ", x_test.shape)

# Obtain betas coefficients that best fit the models with the given data and calculate the respective SSE  
ordinary_tuple = Linear_ordinary_regression(x_train, y_train, x_test)
print('\nSSE for Ordinary regression is:', ordinary_tuple[1])

poly_tuple = polynomial_regression(order, x_train, y_train, x_test)
print("\nSSE for Polynomial regression of order", order, "for the evaluation set is:", poly_tuple[1])

ridge_tuple = ridge(lambda_ridge, x_train, y_train, x_test)
print('\nSSE for Ridge regression with lambda =', lambda_ridge, 'is:', ridge_tuple[1])

lasso_tuple = lasso(lambda_lasso, x_train, y_train, x_test)
print('\nSSE for Lasso regression with lambda =', lambda_lasso, 'is:', lasso_tuple[1])


