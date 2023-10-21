import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from torchvision.transforms import ToPILImage, ToTensor
import keras
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
    x_train_good = y_train[y_train == 1]

    return x_train, y_train, x_test, y_test, x_test_final
from torchvision import transforms
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10)
])

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load data
x_train, y_train, x_test, y_test, x_test_final = load_and_preprocess_data()

# # Combine "bad" and "good" training data
# x_train_combined = np.concatenate((x_train_bad, x_train_good), axis=0)
# y_train_combined = np.concatenate((np.zeros(len(x_train_bad)), np.ones(len(x_train_good))), axis=0)

# Weighted Loss based on the number of samples in each class
weights = []
for label in np.unique(y_train):
    weights.append(len(y_train) / np.count_nonzero(y_train == label))
print("Class weights:", weights)
class_weights = torch.FloatTensor(weights).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# # Further split the training data to create a validation set
# x_train, x_val, y_train, y_val = train_test_split(x_train_combined, y_train_combined, test_size=0.2, random_state=42)

# Random augmentation
aug = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
])

# apply augmentation to the training data where label = 1 
x_train_good_aug = x_train[y_train == 1]
x_train_good_aug = np.array([aug(ToPILImage()(x.transpose(1, 2, 0))) for x in x_train_good_aug])
print(x_train_good_aug.shape)
y_train_good_aug = np.ones(len(x_train_good_aug))
# convert back to numpy array and transpose to the correct shape
x_train_good_aug = np.array([ToTensor()(x).numpy() for x in x_train_good_aug])
print(x_train_good_aug.shape)

# do it once more to get more data
x_train_good_aug2 = x_train[y_train == 1]
x_train_good_aug2 = np.array([aug(ToPILImage()(x.transpose(1, 2, 0))) for x in x_train_good_aug2])
print(x_train_good_aug2.shape)
y_train_good_aug2 = np.ones(len(x_train_good_aug2))
# convert back to numpy array and transpose to the correct shape
x_train_good_aug2 = np.array([ToTensor()(x).numpy() for x in x_train_good_aug2])
print(x_train_good_aug2.shape)

# do it once more to get more data
x_train_good_aug3 = x_train[y_train == 1]
x_train_good_aug3 = np.array([aug(ToPILImage()(x.transpose(1, 2, 0))) for x in x_train_good_aug3])
print(x_train_good_aug3.shape)
y_train_good_aug3 = np.ones(len(x_train_good_aug3))
# convert back to numpy array and transpose to the correct shape
x_train_good_aug3 = np.array([ToTensor()(x).numpy() for x in x_train_good_aug3])
print(x_train_good_aug3.shape)

# do it once more to get more data
x_train_good_aug4 = x_train[y_train == 1]
x_train_good_aug4 = np.array([aug(ToPILImage()(x.transpose(1, 2, 0))) for x in x_train_good_aug4])
print(x_train_good_aug4.shape)
y_train_good_aug4 = np.ones(len(x_train_good_aug4))
# convert back to numpy array and transpose to the correct shape
x_train_good_aug4 = np.array([ToTensor()(x).numpy() for x in x_train_good_aug4])
print(x_train_good_aug4.shape)

# do it once more to get more data
x_train_good_aug5 = x_train[y_train == 1]
x_train_good_aug5 = np.array([aug(ToPILImage()(x.transpose(1, 2, 0))) for x in x_train_good_aug5])
print(x_train_good_aug5.shape)
y_train_good_aug5 = np.ones(len(x_train_good_aug5))
# convert back to numpy array and transpose to the correct shape
x_train_good_aug5 = np.array([ToTensor()(x).numpy() for x in x_train_good_aug5])
print(x_train_good_aug5.shape)


# expand the training data with the augmented data
x_train_combined = np.concatenate((x_train, x_train_good_aug, x_train_good_aug2, x_train_good_aug3, x_train_good_aug4, x_train_good_aug5), axis=0)
y_train_combined = np.concatenate((y_train, y_train_good_aug, y_train_good_aug2, y_train_good_aug3, y_train_good_aug4, y_train_good_aug5), axis=0)
# x_train_combined = np.concatenate((x_train, x_train_good_aug), axis=0)
# y_train_combined = np.concatenate((y_train, y_train_good_aug), axis=0)

# use weights to stratify the data
x_train_combined, x_val, y_train_combined, y_val = train_test_split(x_train_combined, y_train_combined, test_size=0.2, random_state=42)
# ros = RandomOverSampler(random_state=42)
# x_train_combined = x_train_combined.reshape(x_train_combined.shape[0], -1)
# x_train_combined, y_train_combined = ros.fit_resample(x_train_combined, y_train_combined)
# x_train_combined = x_train_combined.reshape(-1, 28, 28, 3).transpose(0, 3, 1, 2)
    # print the number of zeros and 1s in train and in test
print("Train: ", np.unique(y_train_combined, return_counts=True))
print("Val: ", np.unique(y_val, return_counts=True))
print("Test: ", np.unique(y_test, return_counts=True))
# ros = RandomOverSampler()
# x_train_resampled, y_train_resampled = ros.fit_resample(x_train_combined.reshape(x_train_combined.shape[0], -1), y_train_combined)
# x_train_resampled = x_train_resampled.reshape(-1, 3, 28, 28)
# Convert numpy arrays to PyTorch tensors
tensor_x_train = torch.from_numpy(x_train_combined).float().to(device)
tensor_y_train = torch.from_numpy(y_train_combined).long().to(device)
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

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class MediumResNet(nn.Module):
    def __init__(self, num_classes=2):
        super(MediumResNet, self).__init__()
        
        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(16, 2)
        self.layer2 = self._make_layer(32, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, num_classes)

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
# Initialize the CNN and optimizer
model = MediumResNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.00005)

# Training loop with early stopping
# Hyperparameters for tuning
learning_rates = [0.0001, 0.00005, 0.00001, 0.000005, 0.000001, 0.0000005, 0.0000001]  # added learning rates
batch_sizes = [32, 64, 128, 256, 512, 1024, 2048]  # added batch sizes
epochs = 500
best_bac = 0.0
patience = 10  # define how many epochs without improvement before stopping
early_stopping_counter = 0
min_val_loss = float('inf')  # initialize with a high value
results = []
best_bac = 0.0
# Learning Rate Scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
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
        model = MediumResNet().to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
        criterion = nn.CrossEntropyLoss(weight=class_weights)


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
            # Update the scheduler
            scheduler.step(total_val_loss)
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
                # using the current best model predict the labels of the test set
                model.eval()
                total_test_loss = 0
                predictions = []
                true_test_labels = []
                with torch.no_grad():
                    for batch_x, batch_y in test_loader:
                        outputs = model(batch_x)
                        loss = criterion(outputs, batch_y)
                        total_test_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        predictions.extend(predicted.cpu().numpy())
                        true_test_labels.extend(batch_y.cpu().numpy())
                # Calculate metrics for validation set
                bac_test = balanced_accuracy_score(true_test_labels, predictions)
                print("Test BAC:", bac_test)
                cm_test = confusion_matrix(true_test_labels, predictions)
                print("Confusion Matrix for the test set:")
                print(cm_test)

                with open("torch.txt", "a") as f:
                    f.write("\n=====================================")
                    f.write(f"\nBest model with learning rate: {lr}, batch size: {batch_size}, BAC: {bac}")
                    f.write(f"\nConfusion Matrix:\n{cm}")
                    f.write(f"\nTest BAC: {bac_test}")
                    f.write(f"\nTest Confusion Matrix:\n{cm_test}")
                    f.write("\n=====================================")
                

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
