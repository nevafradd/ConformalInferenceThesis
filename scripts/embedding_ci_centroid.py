########################################################
## Embedding-based Conformal Inference
## Loads the three sets of embeddings: train, calibration, test
## Compute centroids on training embeddings
## Compute non-conformity scores on calibration embeddings, find qhat
## Build prediction sets on the test embeddings

## Some sets might be empty - qhat is a distance bound, so if the point is further than qhat
## from all centroids, it will have an empty prediction set

# Ask: "How geometrically similar is this image to a typical member of each
# class in the learned feature space?"
# Previously -> "How confident is the model about each class?"

# Non-conformity score: Distance to class centroid
# || embedding(image) - centroid(true class)||

import torch
import numpy as np

## =============================================================
## CONFIGURATION
## ============================================================

# Choose coverage level: 95%
ALPHA = 0.05
# Choose distance metric.
DISTANCE_METRIC = "euclidean"

print(f"Device: cpu")
print(f"DISTANCE_METRIC: {DISTANCE_METRIC}")
print(f"Coverage target: {(1-ALPHA)*100:.0}%")

#=================================================================
## 1. LOAD EMBEDDINGS

print("\nLoading embeddings...")

train_data = torch.load("train_embeddings.pt")
cal_data = torch.load("cal_embeddings.pt")
test_data = torch.load("test_embeddings.pt")

train_embeddings = train_data["embeddings"] # 60,000 128
cal_embeddings = cal_data["embeddings"] # 1,000 128
test_embeddings = test_data["embeddings"] # 9,000 128

train_labels = train_data["labels"] # 60,000
cal_labels = cal_data["labels"] # 1,000
test_labels = test_data["labels"] # 9,000

print(f"Training embeddings: {train_embeddings.shape}")
print(f"Calibration embeddings: {cal_embeddings.shape}")
print(f"Testing embeddings: {test_embeddings.shape}")

#=======================================================================
## 2. COMPUTE CLASS CENTROIDS
#========================================================================

# For each class, compute mean of all training embeddings
# Gives 10 points in 128-dim space. Each represents the "average" version of
# each digit accordign to the model

# Use training embeddings as it's the biggest set = most stable estimate

NUM_CLASSES = 10
EMBEDDING_DIMENSION = 128
centroids = torch.zeros(NUM_CLASSES, EMBEDDING_DIMENSION)

print("\nComputing centroids...")
for i in range(NUM_CLASSES):
    class_mask = train_labels == i #boolean mask
    class_embeddings = train_embeddings[class_mask]
    centroids[i] = class_embeddings.mean(dim=0)
    print(f" Digit{i}: centroid computed from {class_mask.sum():,} embeddings")

print(f"\nCentroid shape: {centroids.shape}") # should be (10,128)

#==========================================================================
## 3. DISTANCE FUNCTION
#==========================================================================

# computes distances between set of embeddings and all centroids
# INPUT:  embeddings shape (N, 128) — N images to compute distances for
# OUTPUT: distances shape (N, 10)  — distance to each of the 10 centroids
#
# this is used to find qhat, and to build prediction sets

def compute_distances(embeddings, centroids, metric):
    if metric == "euclidean":
        return torch.cdist(embeddings, centroids, p=2)

    elif metric == "cosine":
        embeddings_norm = embeddings / embeddings.norm(dim=1, keepdim=True).clamp(min=1e-8)
        centroids_norm = centroids / centroids.norm(dim=1, keepdim=True).clamp(min=1e-8)

        # cosine similarity is the dot product of normalised vectors
        cosine_sim = embeddings_norm @ centroids_norm.T  # (N, 10)
        return 1 - cosine_sim  # convert similarity to distance

#==============================================================================================
## 4. NON CONFORMITY SCORES ON CALIBRATION SET
#===============================================================================
# non-conformity score = find the distance to the centroid of it's true class
# low = typical example

print("\nComputing calibration nonconformity scores...")

cal_distances = compute_distances(cal_embeddings, centroids, DISTANCE_METRIC)
cal_true_distances = cal_distances.gather(dim=1,
                                          index=cal_labels.unsqueeze(1)
                                          ).squeeze(1)

cal_scores = cal_true_distances

print(f"Calibration nonconformity scores:")
print(f"  Mean:   {cal_scores.mean():.4f}")
print(f"  Median: {cal_scores.median():.4f}")
print(f"  Max:    {cal_scores.max():.4f}")
print(f"  Min:    {cal_scores.min():.4f}")

#======================================================================
## 5. COMPUTE QHAT
#========================================================================
# same 1-alpha quantile method

n = len(cal_scores)
level = np.ceil((n+1) * (1 -ALPHA)) / n
level = min(level, 1.0)

qhat = torch.quantile(cal_scores, level)

print(f"\nConformal threshold (qhat): {qhat:.4f}")
print(f"Interpretation: any class whose centroid is within distance {qhat:.4f}")
print(f"                of the test embedding gets included in the prediction set")

#================================================================================
## 6. BUILD PREDICTION SETS ON THE TEST DATA SET
#=============================================================================

print("\nBuilding prediction sets...")

test_distances = compute_distances(test_embeddings, centroids, DISTANCE_METRIC)
prediction_sets = test_distances <= qhat

print(f"Prediction sets built for {len(prediction_sets):,} test examples")

# =============================================================================
# 7. EVALUATE COVERAGE AND EFFICIENCY
# =============================================================================

# coverage : is true class in the prediction set?
correct_coverage = prediction_sets.gather(
    dim=1,
    index=test_labels.unsqueeze(1)
).squeeze(1)

empirical_coverage = correct_coverage.float().mean().item()

#efficiency: average prediction set size

set_sizes = prediction_sets.sum(dim=1).float()
avg_set_size = set_sizes.mean().item()

print(f"\n{'='*55}")
print(f"RESULTS — EMBEDDING-BASED CONFORMAL INFERENCE")
print(f"{'='*55}")
print(f"Distance metric:    {DISTANCE_METRIC}")
print(f"Target coverage:    {(1 - ALPHA) * 100:.0f}%")
print(f"Empirical coverage: {empirical_coverage * 100:.2f}%")
print(f"Average set size:   {avg_set_size:.3f}")
print(f"{'='*55}")

if empirical_coverage >= (1 - ALPHA):
    print("✓ Coverage guarantee holds")
else:
    print("✗ Coverage guarantee does not hold")

# =============================================================================
# 8. SET SIZE DISTRIBUTION
# =============================================================================

print(f"\nSet size distribution:")
for size in range(0, 11):
    count   = (set_sizes == size).sum().item()
    percent = count / len(set_sizes) * 100
    bar     = "█" * int(percent / 2)
    print(f"  Size {size:2d}: {count:5,} ({percent:5.1f}%) {bar}")

# watch for size 0 sets - images that are far from all centroids

# =============================================================================
# 9. CENTROID DISTANCES — UNDERSTANDING THE EMBEDDING SPACE
# =============================================================================

# which digit classes are close together in the embedding space - these are the pairs
# which will liekly appear toegther in prediction sets

print(f"\nPairwise centroid distances (which digits are close in embedding space):")

centroid_distances = torch.cdist(centroids, centroids, p=2)

#form a table
header = " " +"".join(f" {i:5d}" for i in range(10))
print(header)
for i in range(10):
    row = f" {i} " + "".join(f" {centroid_distances[i,j]:.2f}" for j in range(10))
    print(row)

print("\nSmallest centroid distances (most similar digit pairs):")
# find the closest pairs — exclude diagonal (distance to self = 0)

centroid_distances.fill_diagonal_(float('inf'))
for _ in range(5):
    min_idx = centroid_distances.argmin()
    i, j = min_idx // 10, min_idx % 10
    dist = centroid_distances[i, j].item()
    print(f"  Digit {i} ↔ Digit {j}: distance {dist:.4f}")
    # mask this pair so the next iteration finds the next closest
    centroid_distances[i, j] = float('inf')
    centroid_distances[j, i] = float('inf')

## =============================================================================
## 10. EXAMPLE PREDICTION SETS
# =============================================================================

# rebuild prediction sets with original distances (before diagonal masking)
test_distances = compute_distances(test_embeddings, centroids, DISTANCE_METRIC)
prediction_sets = test_distances <= qhat

print(f"\nExample prediction sets (first 10 test images):")
print(f"{'Example':<10} {'True':<8} {'Prediction set':<22} {'Size':<6} {'Covered'}")
print("-" * 58)
for i in range(10):
    true_label = test_labels[i].item()
    pred_set = prediction_sets[i].nonzero(as_tuple=True)[0].tolist()
    size = len(pred_set)
    covered = true_label in pred_set
    print(f"{i:<10} {true_label:<8} {str(pred_set):<22} {size:<6} {'✓' if covered else '✗'}")


