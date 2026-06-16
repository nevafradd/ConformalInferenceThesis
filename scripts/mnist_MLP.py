# =============================================================================
# MNIST Warmup — Handwritten Digit Classifier
# =============================================================================
# PURPOSE: Learn the PyTorch training loop before applying BERT to your
#          financial documents. Every pattern here (DataLoader, forward pass,
#          loss, backward, optimizer) transfers directly to Stage 2.
#
# SETUP (run once in your PyCharm terminal):
#   pip install torch torchvision
#
# EXPECTED RESULT: ~98% accuracy on the test set after 5 epochs.
# =============================================================================

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# =============================================================================
# 0. CONFIGURATION
# =============================================================================
# Keeping all settings in one place makes experiments easy to tweak.

BATCH_SIZE = 64       # How many images to process at once
LEARNING_RATE = 1e-3  # How big a step the optimizer takes each update
EPOCHS = 5            # How many times to loop through the full training set
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE will be "cpu" on your laptop — that's fine for MNIST.
# On your cloud GPU it will automatically switch to "cuda".

print(f"Using device: {DEVICE}")

# =============================================================================
# 1. LOAD DATA
# =============================================================================
# transforms.ToTensor()     — converts images from pixel values (0-255)
#                             to floats between 0 and 1
# transforms.Normalize(...) — shifts values so the mean is ~0, std is ~1.
#                             This helps training converge faster.
#
# BERT EQUIVALENT: here we transform raw pixels. With BERT you'll use a
# tokenizer to transform raw text into token IDs — same idea.

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
])

# PyTorch will download MNIST automatically the first time (~11MB).
train_dataset = datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# DataLoader slices the dataset into batches and shuffles training data.
# shuffle=True is important — we don't want the model to learn the order.
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

print(f"Training examples: {len(train_dataset):,}")
print(f"Test examples:     {len(test_dataset):,}")

# =============================================================================
# 2. DEFINE THE MODEL
# =============================================================================
# nn.Module is the base class for every PyTorch model.
# __init__  — define your layers
# forward() — describe how data flows through them
#
# Architecture:
#   Input:   28x28 image = 784 pixels (flattened to a 1D vector)
#   Layer 1: 784  → 256 neurons (with ReLU activation)
#   Layer 2: 256  → 128 neurons (with ReLU activation)
#   Output:  128  → 10 neurons  (one score per digit 0-9)
#
# BERT EQUIVALENT: BERT's output is a 768-dimensional vector per token.
# In Stage 2 you'll replace this whole model with BERT + a small linear
# layer on top (the "classifier head").

class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),               # 28x28 image → 784-length vector
            nn.Linear(784, 256),        # first hidden layer
            nn.ReLU(),                  # activation: sets negatives to 0
            nn.Dropout(0.2),            # randomly zeros 20% of neurons (prevents overfitting)
            nn.Linear(256, 128),        # second hidden layer
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10),         # output: 10 scores, one per digit
        )

    def forward(self, x):
        return self.network(x)

model = MNISTClassifier().to(DEVICE)
print(f"\nModel architecture:\n{model}")

# Count parameters so you can see how big the model is
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# =============================================================================
# 3. LOSS FUNCTION AND OPTIMIZER
# =============================================================================
# CrossEntropyLoss — the standard choice for classification.
#   It measures how wrong the model's predictions are.
#   A perfect prediction gives loss ≈ 0. Random guessing gives loss ≈ 2.3.
#
# Adam optimizer — the standard choice for deep learning.
#   It uses the gradients (computed in backward()) to nudge the weights.
#
# BERT EQUIVALENT: you'll use exactly the same loss and optimizer in Stage 2.

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# =============================================================================
# 4. TRAINING LOOP
# =============================================================================
# This is the core pattern that repeats for every ML project you'll ever do.
#
# For each batch:
#   1. Zero out old gradients        — optimizer.zero_grad()
#   2. Forward pass                  — outputs = model(images)
#   3. Compute loss                  — loss = criterion(outputs, labels)
#   4. Backward pass (backprop)      — loss.backward()
#   5. Update weights                — optimizer.step()

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()  # puts model in training mode (enables dropout etc.)
    total_loss = 0
    correct = 0

    for batch_idx, (images, labels) in enumerate(loader):
        # Move data to the same device as the model (CPU or GPU)
        images, labels = images.to(device), labels.to(device)

        # Step 1: clear gradients from the previous batch
        optimizer.zero_grad()

        # Step 2: forward pass — get predictions
        outputs = model(images)           # shape: (batch_size, 10)

        # Step 3: compute how wrong we were
        loss = criterion(outputs, labels)

        # Step 4: backprop — compute gradients
        loss.backward()

        # Step 5: update weights using the gradients
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        predicted = outputs.argmax(dim=1)   # pick the digit with the highest score
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = correct / len(loader.dataset)
    return avg_loss, accuracy

# =============================================================================
# 5. EVALUATION
# =============================================================================
# torch.no_grad() tells PyTorch not to compute gradients — we don't need
# them during evaluation, and skipping them saves memory and time.

def evaluate(model, loader, criterion, device):
    model.eval()  # puts model in evaluation mode (disables dropout)
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = correct / len(loader.dataset)
    return avg_loss, accuracy

# =============================================================================
# 6. RUN TRAINING
# =============================================================================

print("\n" + "="*60)
print("Starting training...")
print("="*60)

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
    test_loss,  test_acc  = evaluate(model, test_loader, criterion, DEVICE)

    print(
        f"Epoch {epoch}/{EPOCHS} | "
        f"Train loss: {train_loss:.4f}, acc: {train_acc:.3f} | "
        f"Test loss:  {test_loss:.4f}, acc: {test_acc:.3f}"
    )

print("\nDone! You should see test accuracy around 0.97-0.98.")
print("Everything above — DataLoader, forward(), loss, backward(), optimizer.step()")
print("— is exactly what you'll reuse when you plug in BERT.")

# =============================================================================
# 7. SAVE THE MODEL (OPTIONAL)
# =============================================================================
# Good habit to get into — you'll need this for your thesis experiments.

torch.save(model.state_dict(), "mnist_model.pth")
print("\nModel saved to mnist_model.pth")

# To reload it later:
#   model = MNISTClassifier()
#   model.load_state_dict(torch.load("mnist_model.pth"))
