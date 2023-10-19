import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.block(x)

class DiscriminatorBlock(nn.Module):
    def __init__(self, in_filters, out_filters, normalize=True, stride=2):
        super(DiscriminatorBlock, self).__init__()
        layers = [nn.Conv2d(in_filters, out_filters, 4, stride=stride, padding=1)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_filters))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class Generator(nn.Module):
    def __init__(self, input_channels, output_channels, num_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolutional block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model.extend([
                nn.Conv2d(in_features, out_features, 4, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ])
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(num_residual_blocks):
            model.append(ResidualBlock(in_features))

        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model.extend([
                nn.ConvTranspose2d(in_features, out_features, 4, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ])
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model.extend([
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_channels, 7),
            nn.Tanh()
        ])

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_channels):
        super(Discriminator, self).__init__()

        model = [
            DiscriminatorBlock(input_channels, 64, normalize=False),
            DiscriminatorBlock(64, 128),
            DiscriminatorBlock(128, 256),
            DiscriminatorBlock(256, 512, stride=1),
            nn.Conv2d(512, 1, 4, padding=1)
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import numpy as np
from sklearn.model_selection import train_test_split
import itertools
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from tqdm import tqdm
def oversample_data(x_data, y_data, target_samples):
    augmented_images = []
    augmented_labels = []
    num_samples_needed = target_samples - len(y_data[y_data == 1])

    while num_samples_needed > 0:
        for x, y in zip(x_data, y_data):
            if y == 1:  # only augment positive examples
                # Randomly choose a transformation: flip, rotate slightly, or add noise
                choice = np.random.choice(['flip', 'rotate', 'noise'])
                if choice == 'flip':
                    augmented_images.append(np.fliplr(x))
                elif choice == 'rotate':
                    angle = np.random.uniform(-10, 10)  # Random angle between -10 and 10 degrees
                    augmented_images.append(np.rot90(x, k=int(angle/90))) 
                elif choice == 'noise':
                    noise = np.random.normal(0, 0.01, x.shape)
                    augmented_images.append(x + noise)

                augmented_labels.append(1)
                num_samples_needed -= 1
                if num_samples_needed == 0:
                    break

    x_data_oversampled = np.concatenate([x_data, np.array(augmented_images)])
    y_data_oversampled = np.concatenate([y_data, np.array(augmented_labels)])
    
    return x_data_oversampled, y_data_oversampled

def get_predictions(model, dataloader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs in dataloader:
            inputs = torch.from_numpy(inputs.reshape(-1, 28, 28 , 3).transpose(0, 3, 1, 2)).float().to(device) / 255.0
            outputs = model(inputs)
            preds = (outputs > 0.5).float().cpu().numpy()
            predictions.extend(preds)
    model.train()
    return np.array(predictions).flatten()

def load_and_preprocess_data():
    x_train_full = np.load("Xtrain_Classification1.npy").reshape(-1, 28, 28 , 3).transpose(0, 3, 1, 2)
    y_train_full = np.load("ytrain_Classification1.npy")
    x_test_final = np.load("Xtest_Classification1.npy").reshape(-1, 28, 28 , 3).transpose(0, 3, 1, 2)

    # split data making sure there are 1 labels in the y_test otherwise redo the split
    while True:
        x_train, x_test, y_train, y_test = train_test_split(x_train_full, y_train_full, test_size=0.2, random_state=42)
        if 1 in y_test:
            break
    x_train, y_train = oversample_data(x_train, y_train, len(y_train[y_train == 0]))  # Setting target_samples to the number of negative examples
    # print the number of zeros and 1s in train and in test
    print("Train: ", np.unique(y_train, return_counts=True))
    print("Test: ", np.unique(y_test, return_counts=True))


    # Split the train data into 'good' and 'bad' based on labels
    x_train_bad = x_train[y_train == 0]
    x_train_good = x_train[y_train == 1]

    return x_train_bad, x_train_good, x_test, y_test, x_test_final

# Parameters
batch_size = 512
epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
x_train_bad, x_train_good, x_test, y_test, x_test_final = load_and_preprocess_data()
# Convert numpy arrays to PyTorch tensors
# Convert numpy arrays to PyTorch tensors
tensor_x_train_bad = torch.from_numpy(x_train_bad).float().to(device) / 255.0
tensor_x_train_good = torch.from_numpy(x_train_good).float().to(device) / 255.0
tensor_x_test = torch.from_numpy(x_test).float().to(device) / 255.0
tensor_x_test_final = torch.from_numpy(x_test_final).float().to(device) / 255.0
# Create DataLoaders
train_loader_A = DataLoader(tensor_x_train_bad, batch_size=batch_size, shuffle=True)
train_loader_B = DataLoader(tensor_x_train_good, batch_size=batch_size, shuffle=True)
# Create DataLoaders for test data
test_loader = DataLoader(tensor_x_test, batch_size=batch_size, shuffle=False)

# Initialize generators and discriminators
G_A2B = Generator(3, 3, 3).to(device)
G_B2A = Generator(3, 3, 3).to(device)
D_A = Discriminator(3).to(device)
D_B = Discriminator(3).to(device)

# Optimizers & Loss
optimizer_G = optim.Adam(itertools.chain(G_A2B.parameters(), G_B2A.parameters()), lr=0.0005, betas=(0.5, 0.999))
optimizer_D_A = optim.Adam(D_A.parameters(), lr=0.0005, betas=(0.5, 0.999))
optimizer_D_B = optim.Adam(D_B.parameters(), lr=0.0005, betas=(0.5, 0.999))

criterion_GAN = nn.MSELoss()
criterion_cycle = nn.L1Loss()

best_bac = 0.0
for epoch in range(epochs):
    print(f"Epoch {epoch}/{epochs}")
    for real_A, real_B in zip(train_loader_A, train_loader_B):
        
        real_A = real_A.to(device)
        real_B = real_B.to(device)
        
        # Generators A2B and B2A
        optimizer_G.zero_grad()
        
        fake_B = G_A2B(real_A)
        loss_GAN_A2B = criterion_GAN(D_B(fake_B), torch.ones_like(D_B(fake_B)))
        
        fake_A = G_B2A(real_B)
        loss_GAN_B2A = criterion_GAN(D_A(fake_A), torch.ones_like(D_A(fake_A)))
        
        # Cycle loss
        recovered_A = G_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0
        
        recovered_B = G_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0
        
        # Total loss
        loss_G = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        loss_G.backward()
        optimizer_G.step()
        
        # Discriminator A
        optimizer_D_A.zero_grad()
        
        loss_D_A_real = criterion_GAN(D_A(real_A), torch.ones_like(D_A(real_A)))
        loss_D_A_fake = criterion_GAN(D_A(fake_A.detach()), torch.zeros_like(D_A(fake_A.detach())))
        
        loss_D_A = (loss_D_A_real + loss_D_A_fake)*0.5
        loss_D_A.backward()
        optimizer_D_A.step()
        
        # Discriminator B
        optimizer_D_B.zero_grad()
        
        loss_D_B_real = criterion_GAN(D_B(real_B), torch.ones_like(D_B(real_B)))
        loss_D_B_fake = criterion_GAN(D_B(fake_B.detach()), torch.zeros_like(D_B(fake_B.detach())))
        
        loss_D_B = (loss_D_B_real + loss_D_B_fake)*0.5
        loss_D_B.backward()
        optimizer_D_B.step()
        
    test_predictions_db = get_predictions(D_B, x_test)  # Use D_B as the classifier
    test_predictions_da = get_predictions(D_A, x_test)  # Use D_A as the classifier
    bac_db = balanced_accuracy_score(y_test, test_predictions_db)
    bac_da = balanced_accuracy_score(y_test, test_predictions_da)
    # show confusion matrix db and da
    print("Confusion Matrix D_B")
    print(confusion_matrix(y_test, test_predictions_db))
    print("Confusion Matrix D_A")
    print(confusion_matrix(y_test, test_predictions_da))
    print(f"G: {loss_G.item():.4f}, Loss D A: {loss_D_A.item():.4f}, BAC_D A: {bac_da:.4f}, Loss D B: {loss_D_B.item():.4f}, BAC_D B: {bac_db:.4f}")

    if bac_db > best_bac:
        best_bac = bac_db
        torch.save(D_B.state_dict(), "D_B.pt")
        print("Saved model")
    elif bac_da > best_bac:
        best_bac = bac_da
        torch.save(D_A.state_dict(), "D_A.pt")
        print("Saved model")

# load the best model
D_B.load_state_dict(torch.load("D_B.pt"))
D_A.load_state_dict(torch.load("D_A.pt"))
# predict x_test_final with D_B and D_a saving as output_d_b and output_d_a
output_d_b = get_predictions(D_B, x_test_final)
output_d_a = get_predictions(D_A, x_test_final)
# save output_d_b and output_d_a as npy files
np.save("output_d_b.npy", output_d_b)
np.save("output_d_a.npy", output_d_a)