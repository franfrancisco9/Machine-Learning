# Import libraries
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso

def compute_SSE(Y_train, model_1, model_2):
    """
    Function to compute SSE for two models.
    Input:
        Y_train: training y values
        model_1: model 1
        model_2: model 2
    Output:
        SSE: SSE of the best of the two models at each point
        SSE_1: SSE of only model 1
        SSE_2: SSE of only model 2
    """
    SSE = 0
    SSE_1 = 0
    SSE_2 = 0

    # Get Y_train_M1 and Y_train_M2 from the two models
    Y_train_M1 = model_1.predict(X_train)
    Y_train_M2 = model_2.predict(X_train)

    # Compute SSE
    for i in range(len(Y_train)):
        SSE_1 += (Y_train[i] - Y_train_M1[i])**2
        SSE_2 += (Y_train[i] - Y_train_M2[i])**2
        if abs(Y_train[i] - Y_train_M1[i]) < abs(Y_train[i] - Y_train_M2[i]):
            SSE += (Y_train[i] - Y_train_M1[i])**2
        else:
            SSE += (Y_train[i] - Y_train_M2[i])**2
    return SSE, SSE_1, SSE_2

# Load the data
X_train = np.load("X_train_regression2.npy")
Y_train = np.load("y_train_regression2.npy")
X_test = np.load("X_test_regression2.npy")

# Train a simple linear regression model to get a baseline SSE
baseline_model = LinearRegression().fit(X_train, Y_train)
# Compute residuals and SSE
residuals_baseline = Y_train - baseline_model.predict(X_train)
SSE_baseline = np.sum(residuals_baseline**2)
print("Baseline SSE:", SSE_baseline)


#================================================================================================
# RESIDUAL SPLITTING
#================================================================================================
# Use model for Regression 1 Task (Lasso with alpha=0.09)
initial_model = Lasso(alpha=0.09).fit(X_train, Y_train)

# Compute the residuals
residuals = Y_train - initial_model.predict(X_train)

# Get the average residuals values and divide into two groups
threshold = np.median(residuals)
low_residual_idx = np.where(residuals <= threshold)[0]
high_residual_idx = np.where(residuals > threshold)[0]

X_train_low_residual = X_train[low_residual_idx]
Y_train_low_residual = Y_train[low_residual_idx]

X_train_high_residual = X_train[high_residual_idx]
Y_train_high_residual = Y_train[high_residual_idx]

# Train two models on the two groups
model_1 = Lasso(alpha=0.09).fit(X_train_low_residual, Y_train_low_residual)
model_2 = Lasso(alpha=0.09).fit(X_train_high_residual, Y_train_high_residual)

# Loop by computing residuals and retraining the models
for _ in range(10):
    residuals_1 = Y_train - model_1.predict(X_train)
    residuals_2 = Y_train - model_2.predict(X_train)
    
    model_1_data_idx = np.where(np.abs(residuals_1) < np.abs(residuals_2))[0]
    model_2_data_idx = np.where(np.abs(residuals_1) >= np.abs(residuals_2))[0]
    
    model_1.fit(X_train[model_1_data_idx], Y_train[model_1_data_idx])
    model_2.fit(X_train[model_2_data_idx], Y_train[model_2_data_idx])

# Compute SSE
SSE, SSE_1, SSE_2 = compute_SSE(Y_train, model_1, model_2)
print("SSE with residual splitting: ", SSE)
print("SSE of model 1: ", SSE_1)
print("SSE of model 2: ", SSE_2)

# Predict on the test set
Y_test_M1 = model_1.predict(X_test)
Y_test_M2 = model_2.predict(X_test)
predictions_residual = np.column_stack((Y_test_M1, Y_test_M2))
np.save("predictions.npy", predictions_residual)


#================================================================================================
# EM ALGORITHM
#================================================================================================
# Initialization for EM
n_samples = X_train.shape[0]
assignments = np.random.choice([0, 1], size=n_samples)

model_1 = Lasso(alpha=0.01).fit(X_train[assignments == 0], Y_train[assignments == 0])
model_2 = Lasso(alpha=0.01).fit(X_train[assignments == 1], Y_train[assignments == 1])

# EM Algorithm
for iteration in range(100):
    residuals_1 = Y_train.reshape(-1) - model_1.predict(X_train)
    residuals_2 = Y_train.reshape(-1) - model_2.predict(X_train)

    likelihood_1 = np.exp(-0.5 * residuals_1**2)
    likelihood_2 = np.exp(-0.5 * residuals_2**2)

    assignments = likelihood_1 > likelihood_2
    indices_1 = np.where(assignments)[0]
    indices_2 = np.where(~assignments)[0]

    model_1 = Lasso(alpha=0.01).fit(X_train[indices_1], Y_train[indices_1])
    model_2 = Lasso(alpha=0.01).fit(X_train[indices_2], Y_train[indices_2])

# Compute SSE for EM
SSE_EM, SSE_1_EM, SSE_2_EM = compute_SSE(Y_train, model_1, model_2)
print("SSE with EM: ", SSE_EM)
print("SSE of model 1 with EM: ", SSE_1_EM)
print("SSE of model 2 with EM: ", SSE_2_EM)

# Predictions for EM
Y_test_M1_EM = model_1.predict(X_test)
Y_test_M2_EM = model_2.predict(X_test)
predictions_em = np.column_stack((Y_test_M1_EM, Y_test_M2_EM))
np.save("predictions_em.npy", predictions_em)

#================================================================================================
# TUNING PART
#================================================================================================
def train_with_EM(X_train, Y_train, model_class, alpha, change_threshold=1e-5, max_iterations=1000):
    """
    Function to train using EM.
    Input:
        X_train: training x values
        Y_train: training y values
        model_class: model class to use
        alpha: alpha value for the model
        change_threshold: convergence threshold for change in likelihood
        max_iterations: maximum number of iterations to prevent endless loops
    Output:
        model_1: model 1
        model_2: model 2
    """
    n_samples = X_train.shape[0]
    assignments = np.random.choice([0, 1], size=n_samples)
    
    if model_class == LinearRegression:
        model_1 = model_class().fit(X_train[assignments == 0], Y_train[assignments == 0])
        model_2 = model_class().fit(X_train[assignments == 1], Y_train[assignments == 1])
    else:
        model_1 = model_class(alpha=alpha).fit(X_train[assignments == 0], Y_train[assignments == 0])
        model_2 = model_class(alpha=alpha).fit(X_train[assignments == 1], Y_train[assignments == 1])

    prev_likelihood_sum = float('-inf')
    for iteration in range(max_iterations):
        residuals_1 = Y_train.reshape(-1) - model_1.predict(X_train)
        residuals_2 = Y_train.reshape(-1) - model_2.predict(X_train)

        likelihood_1 = np.exp(-0.5 * residuals_1**2)
        likelihood_2 = np.exp(-0.5 * residuals_2**2)

        current_likelihood_sum = np.sum(likelihood_1 + likelihood_2)

        # Check convergence
        if abs(current_likelihood_sum - prev_likelihood_sum) < change_threshold:
            break

        prev_likelihood_sum = current_likelihood_sum

        assignments = likelihood_1 > likelihood_2
        indices_1 = np.where(assignments)[0]
        indices_2 = np.where(~assignments)[0]
        if model_class == LinearRegression:
            model_1 = model_class().fit(X_train[indices_1], Y_train[indices_1])
            model_2 = model_class().fit(X_train[indices_2], Y_train[indices_2])
        else:
            model_1 = model_class(alpha=alpha).fit(X_train[indices_1], Y_train[indices_1])
            model_2 = model_class(alpha=alpha).fit(X_train[indices_2], Y_train[indices_2])
    
    return model_1, model_2

def train_with_residuals(X_train, Y_train, model_class, alpha):
    """
    Function to train using residual splitting.
    Input:
        X_train: training x values
        Y_train: training y values
        model_class: model class to use
        alpha: alpha value for the model
    Output:
        model_1: model 1
        model_2: model 2
    """
    if model_class == LinearRegression:
        initial_model = model_class().fit(X_train, Y_train)
    else:
        initial_model = model_class(alpha=alpha).fit(X_train, Y_train)
    
    residuals = Y_train - initial_model.predict(X_train)
    threshold = np.median(residuals)
    
    low_residual_idx = np.where(residuals <= threshold)[0]
    high_residual_idx = np.where(residuals > threshold)[0]
    if model_class == LinearRegression:
        model_1 = model_class().fit(X_train[low_residual_idx], Y_train[low_residual_idx])
        model_2 = model_class().fit(X_train[high_residual_idx], Y_train[high_residual_idx])
    else:
        model_1 = model_class(alpha=alpha).fit(X_train[low_residual_idx], Y_train[low_residual_idx])
        model_2 = model_class(alpha=alpha).fit(X_train[high_residual_idx], Y_train[high_residual_idx])
        
    return model_1, model_2

# Initialize best SSE and best parameters
best_SSE = float('inf')
best_params = None
best_models = None
best_method = None

# Define model classes to explore
model_classes = [LinearRegression, Ridge, Lasso]

# Define alpha values to explore
alphas = np.arange(0.005, 2.005, 0.005)

# Define max iteration values to explore
max_iterations_values = [10, 50, 100, 200]

for model_class in model_classes:
    for alpha in alphas:
        for max_iterations in max_iterations_values:
            # Train using EM
            model_1_em, model_2_em = train_with_EM(X_train, Y_train, model_class, alpha, max_iterations)
            SSE_em, _, _ = compute_SSE(Y_train, model_1_em, model_2_em)
            
            if SSE_em < best_SSE:
                best_SSE = SSE_em
                best_params = (model_class, alpha, max_iterations)
                best_models = (model_1_em, model_2_em)
                best_method = "EM"
                print(f"New Best SSE with EM: {best_SSE}, Model: {model_class.__name__}, Alpha: {alpha}, Max Iterations: {max_iterations}")

            # Train using residual splitting (no change needed here since it's not iterative)
            model_1_res, model_2_res = train_with_residuals(X_train, Y_train, model_class, alpha)
            SSE_res, _, _ = compute_SSE(Y_train, model_1_res, model_2_res)
            
            if SSE_res < best_SSE:
                best_SSE = SSE_res
                best_params = (model_class, alpha)
                best_models = (model_1_res, model_2_res)
                best_method = "Residual Splitting"
                print(f"New Best SSE with Residual Splitting: {best_SSE}, Model: {model_class.__name__}, Alpha: {alpha}")

# Predict with the best model
best_model_1, best_model_2 = best_models
best_Y_test_M1 = best_model_1.predict(X_test)
best_Y_test_M2 = best_model_2.predict(X_test)
best_predict = np.column_stack((best_Y_test_M1, best_Y_test_M2))
np.save("output.npy", best_predict)
print("Best predictions saved!")
print(f"Best Method: {best_method}, Model: {best_params[0].__name__}, Alpha: {best_params[1]}")
