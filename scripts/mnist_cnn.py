# =============================================================================
# MNIST CNN — Convolutional Neural Network for Digit Classification
# =============================================================================
# Applies a CNN to MNIST.
# Different model architecture compared to the MLP.
# The DataLoader, loss, backward, and optimizer pattern is identical.
#
### pip install torch torchvision

# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# =============================================================================
# 0. CONFIGURATION
# =============================================================================

BATCH_SIZE    = 64
LEARNING_RATE = 1e-3
EPOCHS        = 5
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")
#
# =============================================================================
# 1. LOAD DATA
# =============================================================================
# DataLoader is standard, whatever model comes next.
# 60,000 training / 10,000 test split.

transform = transforms.Compose([
    transforms.ToTensor(), #converts image into a PyTorch tensor, rescales pixel values to 0-1
    transforms.Normalize((0.1307,), (0.3081,))
    # centres around 0, gives sd 1. values above are mean and sd of MNIST (available online)
])

# creates the separate training and testing datasets, and saves them in the data folder
# MNIST comes pre-split into train; hence train= TRUE/FALSE
# download = True : checks if files are already there - then doesn't need to download again
# transform = transform attaches preprocessing (lazy loading)
train_dataset = datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

#DataLoader wraps the dataset and shuffles training data
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

#confirms we have 60,000:10,000 in the split. (:, adds the comma in 10,000)
print(f"Training examples: {len(train_dataset):,}")
print(f"Test examples:     {len(test_dataset):,}")

# =============================================================================
# 2. DEFINE THE MODEL
# =============================================================================
# For the CNN we don't flatten the images, keep them as a 28x28 2D grid through
# all the convolutional layers.
# Flatten at the end after spacial feature extraction.

# ARCHITECTURE WALKTHROUGH:
#
#   Input:               1 channel, 28×28 pixels
#                        (1 channel because MNIST is greyscale; RGB would be 3)
#
#   Conv block 1:
#     nn.Conv2d(1, 32, kernel_size=3, padding=1)
#       - 32 filters, each 3×3
#       - padding=1 adds a border of zeros so output stays 28×28
#       - output: 32 feature maps, each 28×28
#     nn.ReLU()          - zeros out negatives, same as MLP
#     nn.MaxPool2d(2)    - 2×2 pooling, halves spatial dims
#       - output: 32 feature maps, each 14×14
#
#   Conv block 2:
#     nn.Conv2d(32, 64, kernel_size=3, padding=1)
#       - 64 filters, each 3×3, applied to the 32 feature maps from block 1
#       - output: 64 feature maps, each 14×14
#     nn.ReLU()
#     nn.MaxPool2d(2)    - halves again
#       - output: 64 feature maps, each 7×7
#
#   Flatten:
#     64 × 7 × 7 = 3136 numbers — finally collapsed to 1D
#
#   Classifier head:
#     nn.Linear(3136, 128) - same idea as MLP hidden layer
#     nn.ReLU()
#     nn.Dropout(0.5)      - higher dropout here as this is the dense section
#     nn.Linear(128, 10)   - 10 output scores, one per digit
#
# BERT outputs a 768-dimensional vector per document.
# classifier head will take 768 inputs instead of 3136.

class MNISTConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Convolutional feature extractor
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,      # greyscale = 1 input channel
                out_channels=32,    # 32 filters → 32 feature maps out
                kernel_size=3,      # each filter is 3×3 pixels
                padding=1           # zero-pad border to keep size 28×28
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)   # 28×28 → 14×14
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,     # takes the 32 feature maps from block 1
                out_channels=64,    # 64 filters → 64 feature maps out
                kernel_size=3,
                padding=1           # keeps size 14×14
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)   # 14×14 → 7×7
        )

        #  Classifier head
        # After two rounds of conv + pooling, each of the 64 feature maps
        # is 7×7. Flattening gives 64 × 7 × 7 = 3136.
        # split into two sections to extract the 128-dimensional embeddings
        self.embedding_head = nn.Sequential(
            nn.Flatten(),               # 64 × 7 × 7 → 3136
            nn.Linear(3136, 128),       # fully connected hidden layer
            nn.ReLU(),
            nn.Dropout(0.5),            # 50% dropout — stronger regularisation
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 10)  # takes the 128-dim vector, ouputs 10 scores
        )
# describes how a batch moves through the network, ie order of layers
    def forward(self, x, return_embeddings=False): # stays False
        # x shape entering: (batch_size, 1, 28, 28)
        x = self.conv_block1(x)   # (batch_size, 32, 14, 14)
        x = self.conv_block2(x)   # (batch_size, 64, 7, 7)
        embedding = self.embedding_head(x) # (batch, 128) , the embedding
        logits = self.classifier(embedding) # (batch, 10) , the prediction

        if return_embeddings:
            return logits, embedding # return both if we call for embeddings
        return logits  # normally, just return logits


model = MNISTConvNet().to(DEVICE)
# prints model structure: includes in and out put dimensions
print(f"\nModel architecture:\n{model}")

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# CNN has much fewer parameters than MLP

# =============================================================================
# 3. LOSS FUNCTION AND OPTIMIZER
# =============================================================================
# Same as MLP — independent of model architecture.

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# =============================================================================
# 4. TRAINING LOOP
# =============================================================================
# Also identical to the MLP version. This is the point — the training loop
# is a universal pattern. Swap the model, keep everything else.

# call this in the training (step 6)
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    # refreshes the batch loss and number of correct predictions at the start of each epoch
    total_loss = 0
    correct    = 0

    # this for loop runs once per batch
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        # images shape: (64, 1, 28, 28) — batch of 64 greyscale(1) 28×28 images
        # The CNN expects this 4D shape
        # Here we don't flatten — the model handles the 2D structure itself.

        optimizer.zero_grad()           #clears gradients from prev batch
        outputs = model(images)           # forward pass (defined before)
        loss    = criterion(outputs, labels) # compare scores to label, output average error
        loss.backward()                   # backpropagation - automatic
        optimizer.step()                  # update weights (Adam)

        total_loss += loss.item() # accumulated loss across batches in the epoch
        predicted   = outputs.argmax(dim=1)
        correct    += (predicted == labels).sum().item() # updates correct prediction count

    return total_loss / len(loader), correct / len(loader.dataset)

# =============================================================================
# 5. EVALUATION
# =============================================================================
# Identical to the MLP version.
# training loop with no learning or dropout - this is the model we apply to test data
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct    = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs   = model(images)
            loss      = criterion(outputs, labels)
            total_loss += loss.item()
            predicted  = outputs.argmax(dim=1)
            correct   += (predicted == labels).sum().item()

    return total_loss / len(loader), correct / len(loader.dataset)

# =============================================================================
# 6. RUN TRAINING
# =============================================================================

print("\n" + "="*60)
print("Starting training...")
print("="*60)

for epoch in range(1, EPOCHS + 1): #python range excludes upper bound
    # training loop
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
    # evaluation
    test_loss,  test_acc  = evaluate(model, test_loader, criterion, DEVICE)

    print(
        f"Epoch {epoch}/{EPOCHS} | "
        f"Train loss: {train_loss:.4f}, acc: {train_acc:.3f} | " #.3f = 3dp
        f"Test  loss: {test_loss:.4f},  acc: {test_acc:.3f}"
    )

print("\nDone!")
print(f"Final test accuracy: {test_acc:.4f}")
print("You should be seeing ~0.99 — better than the MLP's ~0.98.")
print("The gain comes entirely from the CNN respecting spatial structure.")

# =============================================================================
# 7. SAVE SOFTMAX SCORES FOR CONFORMAL INFERENCE
# =============================================================================
#
# save the test set scores to disk so the CI script can load them without
# retraining the model.


print("\nSaving softmax scores for conformal inference...")

model.eval()
all_probs  = []   # softmax probabilities, shape: (N, 10)
all_labels = []   # true labels,           shape: (N,)

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        logits = model(images)                        # raw scores
        probs  = F.softmax(logits, dim=1)             # probabilities
        all_probs.append(probs.cpu())
        all_labels.append(labels)

all_probs  = torch.cat(all_probs,  dim=0)   # (10000, 10)
all_labels = torch.cat(all_labels, dim=0)   # (10000,)

torch.save({
    "probs":  all_probs,
    "labels": all_labels
}, "mnist_cnn_scores.pt")

print("Scores saved to mnist_cnn_scores.pt")
print("These will be loaded by the conformal inference script.")

# Save the model weights too
torch.save(model.state_dict(), "mnist_cnn.pth")
print("Model saved to mnist_cnn.pth")

