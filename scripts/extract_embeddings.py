#######################################################
## Loads the trained model, extracts and saves training, calibration, and
## test embeddings
#######################################################

# The trained CNN is used as a feature extractor
# Extracts 128-dimensional embeddings from the layer just before the final
# classification layer for 3 data sets
# Training set: 60,000 images (computes the class centroids)
# Calibration set: 1,000 images (computes conformal threshold)
# Test set: 9,000 images (builds prediction sets)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets

## 0. CONFIGURATION

BATCH_SIZE = 64
N_CALIBRATION = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

## 1. DEFINE UPDATED ARCHITECTURE (split classifier head)
#=======================================================================
#old:
# self.classifier = nn.Sequential(Flatten, Linear(3136,128),
#                             ReLU, Dropout, Linear(128,10))
#new:
# self.embedding_head = nn.Sequential(Flatten, Linear(3136,128),
#                                   ReLU, Dropout)
#self.classifier  = nn.Sequential(Linear(128,10))
#
# forward() returns embeddings and logits when return_embeddings= True

# Model weights are saved, not the model structure, give model structure here

class MNISTConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Convolutional feature extractor
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 28×28 → 14×14
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 14×14 → 7×7
        )

        # Embedding head
        # Produces the 128-dimensional embedding vector.
        self.embedding_head = nn.Sequential(
            nn.Flatten(),  # 64 × 7 × 7 → 3136
            nn.Linear(3136, 128),  # → 128-dimensional embedding
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # Classifier head
        # Takes the 128-dim embedding and produces 10 class scores.
        self.classifier = nn.Sequential(
            nn.Linear(128, 10)  # → 10 class scores
        )

    def forward(self, x, return_embeddings=False):
        # x shape: (batch_size, 1, 28, 28)
        x = self.conv_block1(x)  # → (batch_size, 32, 14, 14)
        x = self.conv_block2(x)  # → (batch_size, 64, 7, 7)
        embedding = self.embedding_head(x)  # → (batch_size, 128)
        logits = self.classifier(embedding)  # → (batch_size, 10)

        if return_embeddings:
            return logits, embedding  # return both for extraction
        return logits  # return only logits during training

 #==================================================================
# ## 2. LOAD TRAINED WEIGHTS FROM mnist_cnn.py
#==================================================================
#Instantiate the model = run the model so it actually exists (?)
# Load weights

model = MNISTConvNet().to(DEVICE)
model.load_state_dict(torch.load("mnist_cnn.pth", map_location=DEVICE))
model.eval() #no dropout - weights are frozen - no learning

print("Model loaded")

#=============================================================
# 3. LOAD DATA
#===================================================================


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))]
)

train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

# split test into calibration and test sets
# use same random seed for same split

torch.manual_seed(42)
indices = torch.randperm(len(test_dataset))
cal_indices = indices[:N_CALIBRATION]
test_indices = indices[N_CALIBRATION:]

cal_dataset = Subset(test_dataset, cal_indices)
test_subset = Subset(test_dataset, test_indices)

# Dataloaders - shuffle = False as order doesn't matter.

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
cal_loader = DataLoader(cal_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train set size: {len(train_dataset):,} images")
print(f"Calibration set size: {len(cal_dataset):,} images")
print(f"Test set size: {len(test_subset):,} images")

#===================================================================
## 4. EXTRACTION FUNCTION
#===================================================================

# function that runs a DataLoader through the model, collects embeddings and labels

def extract_embeddings(loader, model, device):
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            _,embeddings = model(images, return_embeddings=True)
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())
    embeddings = torch.cat(all_embeddings, dim=0)
    labels = torch.cat(all_labels, dim=0)

    return embeddings, labels

#==================================================================
## 5. EXTRACT AND SAVE EMBEDDINGS
#==================================================================

print("\nExtracting embeddings...")
train_embeddings, train_labels = extract_embeddings(train_loader, model, DEVICE)
print(f" Shape: {train_embeddings.shape}") # 60,000, 128

cal_embeddings, cal_labels = extract_embeddings(cal_loader, model, DEVICE)
print(f" Shape: {cal_embeddings.shape}") # 1000, 128

test_embeddings, test_labels = extract_embeddings(test_loader, model, DEVICE)
print(f" Shape: {test_embeddings.shape}") # 9000, 128

# save sets to disk to call them for conformal inference

torch.save({"embeddings": train_embeddings, "labels": train_labels}, "train_embeddings.pt")
torch.save({"embeddings": cal_embeddings, "labels": cal_labels}, "cal_embeddings.pt")
torch.save({"embeddings": test_embeddings, "labels": test_labels}, "test_embeddings.pt")

print("\nSaved:")
print("  train_embeddings.pt")
print("  cal_embeddings.pt")
print("  test_embeddings.pt")


##===============================================================
## 6. CHECK
##================================================================

# Verify embeddings look sensible before passing on to CI script
# should be 128-dim , and non negative (ReLU)

print(f"  Min value:  {train_embeddings.min():.4f}  (should be >= 0 due to ReLU)")
print(f"  Max value:  {train_embeddings.max():.4f}")
print(f"  Mean value: {train_embeddings.mean():.4f}")
print(f"  Any NaN?    {torch.isnan(train_embeddings).any()}")  # should be False

print("\nLabel distribution in training set:")
for digit in range(10):
    count = (train_labels == digit).sum().item()
    percent = count / len(train_labels) * 100
    print(f"  Digit {digit}: {count:,} ({percent:.1f}%)")
# MNIST is roughly balanced — each digit should be about 10% of the data

print("\nDone. Ready to run embedding_ci.py")