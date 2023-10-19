import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

def load_and_preprocess_data():
    x_train_full = np.load("Xtrain_Classification1.npy").reshape(-1, 28, 28 , 3).transpose(0, 3, 1, 2)
    y_train_full = np.load("ytrain_Classification1.npy")
    x_test_final = np.load("Xtest_Classification1.npy").reshape(-1, 28, 28 , 3).transpose(0, 3, 1, 2)

    # split data making sure there are 1 labels in the y_test otherwise redo the split
    while True:
        x_train, x_test, y_train, y_test = train_test_split(x_train_full, y_train_full, test_size=0.2, random_state=42)
        if 1 in y_test:
            break

    # print the number of zeros and 1s in train and in test
    print("Train: ", np.unique(y_train, return_counts=True))
    print("Test: ", np.unique(y_test, return_counts=True))


    # Split the train data into 'good' and 'bad' based on labels
    x_train_bad = x_train[y_train == 0]
    x_train_good = x_train[y_train == 1]

    return x_train_bad, x_train_good, x_test, y_test, x_test_final

device = "cuda" if torch.cuda.is_available() else "cpu"
# Hyperparameters for tuning
learning_rates = [0.05, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
batch_sizes = [16, 32, 64, 128, 256, 512]  # added batch sizes
epochs = 100
patience = 20  # define how many epochs without improvement before stopping

# Load data
x_train_bad, x_train_good, x_test, y_test, x_test_final = load_and_preprocess_data()

# Combine "bad" and "good" training data
x_train_combined = np.concatenate((x_train_bad, x_train_good), axis=0)
y_train_combined = np.concatenate((np.zeros(len(x_train_bad)), np.ones(len(x_train_good))), axis=0)

# Further split the training data to create a validation set
x_train, x_val, y_train, y_val = train_test_split(x_train_combined, y_train_combined, test_size=0.2, random_state=42)

# Oversampling
ros = RandomOverSampler()
x_train_resampled, y_train_resampled = ros.fit_resample(x_train.reshape(x_train.shape[0], -1), y_train)
x_train_resampled = x_train_resampled.reshape(-1, 3, 28, 28)
# Convert numpy arrays to PyTorch tensors
tensor_x_train = torch.from_numpy(x_train_resampled).float().to(device)
tensor_y_train = torch.from_numpy(y_train_resampled).long().to(device)
tensor_x_val = torch.from_numpy(x_val).float().to(device)
tensor_y_val = torch.from_numpy(y_val).long().to(device)
tensor_x_test = torch.from_numpy(x_test).float().to(device)
tensor_y_test = torch.from_numpy(y_test).long().to(device)


# Define a simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Initialize the CNN and optimizer
model = SimpleCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop with early stopping
best_bac = 0.0
patience = 20  # define how many epochs without improvement before stopping
early_stopping_counter = 0
min_val_loss = float('inf')  # initialize with a high value
results = []
best_bac = 0.0
# Hyperparameter tuning loop
for lr in learning_rates:
    for batch_size in batch_sizes:
        
        # Create DataLoaders
        train_dataset = TensorDataset(tensor_x_train, tensor_y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = TensorDataset(tensor_x_val, tensor_y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        test_dataset = TensorDataset(tensor_x_test, tensor_y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Initialize the CNN and optimizer
        model = SimpleCNN().to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        print(f"\nTraining with learning rate: {lr}, batch size: {batch_size}")
        
        # Training loop with early stopping
        early_stopping_counter = 0
        min_val_loss = float('inf')
        for epoch in range(epochs):
            # Training phase
            model.train()
            total_train_loss = 0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            # Validation phase
            model.eval()
            total_val_loss = 0
            predictions = []
            true_val_labels = []
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    total_val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    predictions.extend(predicted.cpu().numpy())
                    true_val_labels.extend(batch_y.cpu().numpy())

            # Calculate metrics for validation set
            bac = balanced_accuracy_score(true_val_labels, predictions)
            cm = confusion_matrix(true_val_labels, predictions)

            # Check if we need to save the model
            if bac > best_bac:
                best_bac = bac
                torch.save(model.state_dict(), "best_model.pt")
                print("Model saved with BAC:", bac)
                print("Confusion Matrix for the best model:")
                print(cm)

            # Check early stopping conditions
            if total_val_loss >= min_val_loss:
                early_stopping_counter += 1
            else:
                early_stopping_counter = 0
                min_val_loss = total_val_loss

            if early_stopping_counter >= patience:
                print("Early stopping after {} epochs without improvement".format(patience))
                break

            # Printing epoch details
            print(f"Epoch {epoch}/{epochs}, Train Loss: {total_train_loss/len(train_loader):.4f}, Val Loss: {total_val_loss/len(val_loader):.4f}")
                # Append results to results list
        results.append({
            'learning_rate': lr,
            'batch_size': batch_size,
            'best_bac': best_bac,
            'min_val_loss': min_val_loss
        })       

# Compare results and find the best combination
best_result = max(results, key=lambda x: x['best_bac'])
print(f"\nBest combination found: learning rate {best_result['learning_rate']}, batch size {best_result['batch_size']}.")
print(f"Best balanced accuracy: {best_result['best_bac']}, Min validation loss: {best_result['min_val_loss']}")
