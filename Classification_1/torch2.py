import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from torchvision.transforms import ToPILImage, ToTensor

to_pil = ToPILImage()
to_tensor = ToTensor()

class AugmentedDataset(Dataset):
    def __init__(self, x_data, y_data, transform=None):
        self.x_data = x_data
        self.y_data = y_data
        self.transform = transform

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        sample = self.x_data[idx]
        if self.transform:
            sample = to_pil(sample) 
            sample = self.transform(sample)
            sample = to_tensor(sample)
        return sample, self.y_data[idx]

def load_and_preprocess_data():
    x_train_full = np.load("Xtrain_Classification1.npy").reshape(-1, 28, 28 , 3).transpose(0, 3, 1, 2)
    y_train_full = np.load("ytrain_Classification1.npy")
    x_test_final = np.load("Xtest_Classification1.npy").reshape(-1, 28, 28 , 3).transpose(0, 3, 1, 2)

    # split data making sure there are 1 labels in the y_val otherwise redo the split
    while True:
        x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.2, random_state=42)
        if 1 in y_val:
            break

    # print the number of zeros and 1s in train and in val
    print("Train: ", np.unique(y_train, return_counts=True))
    print("Val: ", np.unique(y_val, return_counts=True))

    return x_train, y_train, x_val, y_val, x_test_final                                                                                                                                                                                                                                                                                                             
from torchvision import transforms

# Define the augmentation transform
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
])

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load data
x_train, y_train, x_val, y_val, x_test_final = load_and_preprocess_data()
# Convert numpy arrays to PyTorch tensors
tensor_x_train = torch.from_numpy(x_train).float().to(device)
tensor_y_train = torch.from_numpy(y_train).long().to(device)

# Create the augmented dataset for training
train_dataset = TensorDataset(tensor_x_train, tensor_y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Convert validation data to PyTorch tensors
tensor_x_val = torch.from_numpy(x_val).float().to(device)
tensor_y_val = torch.from_numpy(y_val).long().to(device)

# Weighted Loss for training
weights = [len(y_train) / np.sum(y_train == i) for i in np.unique(y_train)]
class_weights = torch.FloatTensor(weights).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)





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
optimizer = optim.Adam(model.parameters(), lr=0.0000000005)

# Training loop with early stopping
# Hyperparameters for tuning
learning_rates = [ 0.0005, 0.0001, 0.00005, 0.00001, 0.000005, 0.000001, 0.0000005, 0.0000001]
batch_sizes = [32, 64, 128, 256, 512, 1024, 2048]  # added batch sizes
epochs = 1000
best_bac = 0.0
patience = 20  # define how many epochs without improvement before stopping
early_stopping_counter = 0
min_val_loss = float('inf')  # initialize with a high value
results = []
best_bac = 0.0
# Learning Rate Scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, verbose=True)
# Hyperparameter tuning loop
for lr in learning_rates:
    for batch_size in batch_sizes:
        
        # Create DataLoaders
        train_dataset = TensorDataset(tensor_x_train, tensor_y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = TensorDataset(tensor_x_val, tensor_y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Initialize the CNN and optimizer
        model = MediumResNet().to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1, verbose=True)
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
                batch_x, batch_y = batch_x.to(device), batch_y.to(device) 
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
                # print the predictions for the test set
                model.eval()

                with open("torch.txt", "a") as f:
                    f.write("\n=====================================")
                    f.write(f"\nBest model with learning rate: {lr}, batch size: {batch_size}, BAC: {bac}")
                    f.write(f"\nConfusion Matrix:\n{cm}")
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