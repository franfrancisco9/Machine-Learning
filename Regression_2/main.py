import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error

# Load the data
X_train = np.load("X_train_regression2.npy")
Y_train = np.load("y_train_regression2.npy")
X_test = np.load("X_test_regression2.npy")

# 1. Train a single linear regression model on the entire dataset
initial_model = LinearRegression().fit(X_train, Y_train)

# 2. Compute residuals
residuals = Y_train - initial_model.predict(X_train)

# 3. Split the data based on residuals
# Here, I'm using the median as a threshold. This can be tuned.
threshold = np.median(residuals)
low_residual_idx = np.where(residuals <= threshold)[0]
high_residual_idx = np.where(residuals > threshold)[0]

X_train_low_residual = X_train[low_residual_idx]
Y_train_low_residual = Y_train[low_residual_idx]

X_train_high_residual = X_train[high_residual_idx]
Y_train_high_residual = Y_train[high_residual_idx]

# 4. Train separate models
model_1 = Ridge(alpha=1.0).fit(X_train_low_residual, Y_train_low_residual)
model_2 = Ridge(alpha=1.0).fit(X_train_high_residual, Y_train_high_residual)

# 5. Iterative Refinement
# Note: This is a very basic iterative refinement. In practice, you might want more conditions to break the loop.
for _ in range(100):  # run for 10 iterations or until a certain convergence criterion is met
    residuals_1 = Y_train - model_1.predict(X_train)
    residuals_2 = Y_train - model_2.predict(X_train)
    
    model_1_data_idx = np.where(np.abs(residuals_1) < np.abs(residuals_2))[0]
    model_2_data_idx = np.where(np.abs(residuals_1) >= np.abs(residuals_2))[0]
    
    model_1.fit(X_train[model_1_data_idx], Y_train[model_1_data_idx])
    model_2.fit(X_train[model_2_data_idx], Y_train[model_2_data_idx])

# calculate SSE for each model on the training set assume the best 
# result in each iteration when calculating the SSE
SSE = 0
Y_train_M1 = model_1.predict(X_train)
Y_train_M2 = model_2.predict(X_train)
for i in range(len(Y_train)):
    if abs(Y_train[i] - Y_train_M1[i]) < abs(Y_train[i] - Y_train_M2[i]):
        SSE += (Y_train[i] - Y_train_M1[i])**2
    else:
        SSE += (Y_train[i] - Y_train_M2[i])**2
print("SSE with residual splitting: ", SSE)




# 6. Predict on the test set
Y_test_M1 = model_1.predict(X_test)
Y_test_M2 = model_2.predict(X_test)

# Stacking predictions for submission or further analysis
predictions = np.column_stack((Y_test_M1, Y_test_M2))

# Save predictions
np.save("predictions.npy", predictions)

print("Predictions saved!")

# Initialization
n_samples = X_train.shape[0]
assignments = np.random.choice([0, 1], size=n_samples)

model_1 = LinearRegression().fit(X_train[assignments == 0], Y_train[assignments == 0])
model_2 = LinearRegression().fit(X_train[assignments == 1], Y_train[assignments == 1])

for iteration in range(10):  # e.g., 100 iterations or some convergence criterion
    # E-step: Calculate likelihood under each model and assign points accordingly
    residuals_1 = Y_train.reshape(-1) - model_1.predict(X_train)
    residuals_2 = Y_train.reshape(-1) - model_2.predict(X_train)

    likelihood_1 = np.exp(-0.5 * residuals_1**2)
    likelihood_2 = np.exp(-0.5 * residuals_2**2)

    assignments = likelihood_1 > likelihood_2
    indices_1 = np.where(assignments)[0]

    # M-step: Refit models based on current assignments
    model_1 = LinearRegression().fit(X_train[indices_1], Y_train[indices_1])
    model_2 = LinearRegression().fit(X_train[~indices_1], Y_train[~indices_1])

# calculate SSE for each model on the training set assume the best 
# result in each iteration when calculating the SSE
SSE = 0
Y_train_M1 = model_1.predict(X_train)
Y_train_M2 = model_2.predict(X_train)
for i in range(len(Y_train)):
    if abs(Y_train[i] - Y_train_M1[i]) < abs(Y_train[i] - Y_train_M2[i]):
        SSE += (Y_train[i] - Y_train_M1[i])**2
    else:
        SSE += (Y_train[i] - Y_train_M2[i])**2
print("SSE with EM: ", SSE)

# Predictions
Y_test_M1 = model_1.predict(X_test)
Y_test_M2 = model_2.predict(X_test)

predictions = np.column_stack((Y_test_M1, Y_test_M2))
np.save("predictions_em.npy", predictions)

print("EM Predictions saved!")
