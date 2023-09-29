# Import libraries
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
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
alpha_ridge = 1.89 # Regularization term for Ridge regression
alpha_lasso = 0.096 # Regularization term for Lasso regression

print("\nData shape of vector of features data and vector of outcomes data for training: ", x_train.shape, y_train.shape)
print("\nData shape of vector of features for prediction: ", x_test.shape)

# Obtain betas coefficients that best fit the models with the given data and calculate the respective SSE  
ordinary_tuple = Linear_ordinary_regression(x_train, y_train, x_test)
print('\nSSE for Ordinary regression is:', ordinary_tuple[1][0])

poly_tuple = polynomial_regression(order, x_train, y_train, x_test)
print("\nSSE for Polynomial regression of order", order, "for the evaluation set is:", poly_tuple[1][0])

ridge_tuple = ridge(alpha_ridge, x_train, y_train, x_test)
print('\nSSE for Ridge regression with Alpha =', alpha_ridge, 'is:', ridge_tuple[1][0])

lasso_tuple = lasso(alpha_lasso, x_train, y_train, x_test)     
print('\nSSE for Lasso regression with Alpha =', alpha_lasso, 'is:', lasso_tuple[1][0])


# ------------------------------------------------------------------------------------------------------------------------------------ #
# At this point we already know which model better minimizes the SSE and better adjusts to the training data.
# But we cannot guarentee that this model is robust for new data, for example, the polynomial regression may suffer from overfitting.
# For this reason, we are going to consider a third set, the validation set.
# Separate known data into training set and validation set in order to check hyperparameters or overfiting, for example.
test_size = 0.3
(x_train_new, x_validation, y_train_new, y_validation) = train_test_split(x_train, y_train, test_size=test_size, shuffle=False) 

ordinary_tuple = Linear_ordinary_regression(x_train_new, y_train_new, x_validation)
# Calculate SSE with the validation set
y_ord_test = ordinary_tuple[2]
SSE = (np.linalg.norm(y_validation - y_ord_test))**2
print("\nSSE for Ordinary regression, for the validation set is:", SSE)

poly_tuple = polynomial_regression(order, x_train_new, y_train_new, x_validation)
# Calculate SSE with the validation set
y_poly_test = poly_tuple[2]
SSE = (np.linalg.norm(y_validation - y_poly_test))**2
print("\nSSE for Polynomial regression of order", order, "for the validation set is:", SSE)

ridge_tuple = ridge(alpha_ridge, x_train_new, y_train_new, x_validation)
# Calculate SSE with the validation set
y_ridge_test = ridge_tuple[2]
SSE = (np.linalg.norm(y_validation - y_ridge_test))**2
print("\nSSE for Ridge regression with Alpha =", alpha_ridge, "for the validation set is:", SSE)

lasso_tuple = lasso(alpha_lasso, x_train_new, y_train_new, x_validation)
# Calculate SSE with the validation set
y_lasso_test = lasso_tuple[2]
SSE = (np.linalg.norm(y_validation - y_lasso_test))**2
print("\nSSE for Lasso regression with Alpha =", alpha_lasso, "for the validation set is:", SSE)


# Tunning hyperparameters for lasso and ridge which appear above
# Average SSE comparison with cross-validation using different values of alpha for Ridge regression and Lasso regression 

# Validation for different test sizes and different Alphas:
alpha_ridge = np.arange(0.001, 10, 0.001) # alpha array to tune for ridge
alpha_lasso = np.arange(0.001, 10, 0.001) # alpha array to tune for lasso
test_size_vector = np.arange(0.065, 0.35, 0.01)
SSE_split_ridge = np.zeros(len(alpha_ridge))
SSE_split_lasso = np.zeros(len(alpha_lasso))
SSE_temp = 0
for i in range(len(alpha_ridge)):
  for test_size in test_size_vector:

      # Split Training set into new Training data and a Validation set
      (x_train_new, x_validation, y_train_new, y_validation) = train_test_split(x_train, y_train, test_size=test_size, shuffle=True, random_state=42) 

      ridge_tuple = ridge(alpha_ridge[i], x_train_new, y_train_new, x_validation)
      # Calculate SSE with the validation set
      y_ridge_test = ridge_tuple[2]
      SSE_split_ridge[i] += (np.linalg.norm(y_validation - y_ridge_test))**2

  SSE_split_ridge[i] = SSE_split_ridge[i]/len(test_size_vector)


for i in range(len(alpha_lasso)):
  for test_size in test_size_vector:

      # Split Training set into new Training data and a Validation set
      (x_train_new, x_validation, y_train_new, y_validation) = train_test_split(x_train, y_train, test_size=test_size, shuffle=True, random_state=42) 

      lasso_tuple = lasso(alpha_lasso[i], x_train_new, y_train_new, x_validation)
      # Calculate SSE with the validation set
      y_lasso_test = lasso_tuple[2]
      SSE_split_lasso[i] += (np.linalg.norm(y_validation - y_lasso_test))**2
  SSE_split_lasso[i] = SSE_split_lasso[i]/len(test_size_vector)

# Validation using Cross-Validations with K = Kfolds 
Kfolds = 5
SSE_validation_ridge = np.zeros(len(alpha_ridge))
SSE_validation_lasso = np.zeros(len(alpha_lasso))
SSE_min = 1000
SSE_temp = 0

kf = KFold(n_splits=Kfolds, random_state=None, shuffle=False)
for i in range(len(alpha_ridge)):
    for train_new_index, test_validation_index in kf.split(x_train):

        x_train_new, x_validation = x_train[train_new_index], x_train[test_validation_index]
        y_train_new, y_validation = y_train[train_new_index], y_train[test_validation_index]

        ridge_tuple = ridge(alpha_ridge[i], x_train_new, y_train_new, x_validation)

        # Calculate SSE with the validation set
        y_ridge_test = ridge_tuple[2]
        for k in range(y_validation.shape[0]):
            SSE_temp += (y_validation[k] - y_ridge_test[k][0])**2

    SSE_validation_ridge[i] = SSE_temp/Kfolds
    SSE_temp = 0

    if(SSE_validation_ridge[i] < SSE_min):
        best_alpha_ridge = alpha_ridge[i]
        SSE_min = SSE_validation_ridge[i]

SSE_min = 1000
for i in range(len(alpha_lasso)):
    for train_new_index, test_validation_index in kf.split(x_train):

        x_train_new, x_validation = x_train[train_new_index], x_train[test_validation_index]
        y_train_new, y_validation = y_train[train_new_index], y_train[test_validation_index]

        lasso_tuple = lasso(alpha_lasso[i], x_train_new, y_train_new, x_validation)

        # Calculate SSE with the validation set
        y_lasso_test = lasso_tuple[2]
        for k in range(y_validation.shape[0]):
            SSE_temp += (y_validation[k] - y_lasso_test[k])**2
    
    SSE_validation_lasso[i] = SSE_temp/Kfolds
    SSE_temp = 0

    if(SSE_validation_lasso[i] < SSE_min):
        best_alpha_lasso = alpha_lasso[i]
        SSE_min = SSE_validation_lasso[i]

# Cross-validation for different values of alpha and random shuffling of data 
SSE_cv_shuffle_ridge = np.zeros(len(alpha_ridge))
SSE_cv_shuffle_lasso = np.zeros(len(alpha_lasso))

for i in range(len(alpha_ridge)):
  SSE_cv_shuffle_ridge[i]  = Kfolds_ridge_regression(Kfolds, x_train, y_train, best_alpha_ridge, 100)

for i in range(len(alpha_lasso)):
  SSE_cv_shuffle_lasso[i]  = Kfolds_lasso_regression(Kfolds, x_train, y_train, best_alpha_lasso, 100)

# Plot results

plot_ridge = plt.figure(1)
plt.plot(alpha_ridge, SSE_validation_ridge)

# Plot axes labels and show the plot
plt.xlabel('Alpha [Ridge]')
plt.ylabel('SSE')
plt.title('Average SSE computed after Cross Validation with K = {} folds'.format(Kfolds))
plt.draw()

plot_lasso = plt.figure(2)
plt.plot(alpha_lasso, SSE_validation_lasso)

# Plot axes labels and show the plot
plt.xlabel('Alpha [Lasso]')
plt.ylabel('SSE')
plt.title('Average SSE computed after Cross Validation with K = {} folds'.format(Kfolds))
plt.draw()

# Compare in same plot

plot_ridge = plt.figure(7)
plt.plot(alpha_ridge, SSE_validation_ridge)

# Plot axes labels and show the plot
plt.xlabel('Alpha')
plt.ylabel('SSE')
plt.title('Ridge vs Lasso SSE computed after Cross Validation with K = {} folds'.format(Kfolds))
plt.draw()

plot_lasso = plt.figure(7)
plt.plot(alpha_lasso, SSE_validation_lasso)
plt.draw()

plot_lasso = plt.figure(3)
plt.plot(alpha_ridge, SSE_cv_shuffle_ridge)

# Plot axes labels and show the plot
plt.xlabel('Alpha [Ridge]')
plt.ylabel('SSE')
plt.title('Average SSE computed after Cross Validation with shuffling for K = {} folds'.format(Kfolds))
plt.draw()

plot_lasso = plt.figure(4)
plt.plot(alpha_lasso, SSE_cv_shuffle_lasso)

# Plot axes labels and show the plot
plt.xlabel('Alpha [Lasso]')
plt.ylabel('SSE')
plt.title('Average SSE computed after Cross Validation with shuffling for K = {} folds'.format(Kfolds))
plt.draw()

plot_lasso = plt.figure(5)
plt.plot(alpha_ridge, SSE_split_ridge)

# Plot axes labels and show the plot
plt.xlabel('Alpha [Ridge]')
plt.ylabel('SSE')
plt.title('Average SSE computed after Cross Validation with different test sizes')
plt.draw()

plot_lasso = plt.figure(6)
plt.plot(alpha_lasso, SSE_split_lasso)

# Plot axes labels and show the plot
plt.xlabel('Alpha [Lasso]')
plt.ylabel('SSE')
plt.title('Average SSE computed after Cross Validation with different test sizes')
plt.draw()

# ------------------------------------------- Compare the models with tuned hyperparameters using cross-validation ------------------------------------------- #

SSE_ord = Kfolds_linear_ordinary_regression(Kfolds, x_train, y_train, 100)
SSE_ridge = Kfolds_ridge_regression(Kfolds, x_train, y_train, best_alpha_ridge, 100)
SSE_lasso = Kfolds_lasso_regression(Kfolds, x_train, y_train, best_alpha_lasso, 100)

print('SSE ordinary:', SSE_ord, '\nSSE Ridge:', SSE_ridge, '\nSSE Lasso:', SSE_lasso)

print("Best alpha for Ridge regression:", best_alpha_ridge)
print("Best alpha for Lasso regression:", best_alpha_lasso)

# Submision - Best model with tuned hyperparameters


ridge_tuple = ridge(best_alpha_ridge, x_train, y_train, x_test)
y_submit = ridge_tuple[2]

np.save('output', y_submit)

plt.show()
