# Second run through of embedding based CI on MNIST.
# Rather than looking at the distance of each embedding from it's true class centroid, this looks at
# the distance from the point to it's nearest neighbour in the true class
# DNN = distance to nearest neighbour
# more sensitive to local density - especially if we have sub clusters within a cluster
# centroid method works best for spherical clusters

import torch
import numpy as np

# Configuration, choose coverage and distance metric

ALPHA = 0.05
DISTANCE_METRIC = "euclidean"

# choose how many neighbours to consider.
# K = 1 : most sensitive to local structure, can be noisy if the closest example is an outlier
# K = 5 : average distance to 5 closest examples, more robust but behaves more like a local density estimate
# compare both
K = 5

print(f"Distance metric: {DISTANCE_METRIC}")
print(f"K (neighbours considered): {K}")
print(f"Coverage target: {(1 - ALPHA) * 100:.0f}%")

#====================================================================
## 1. LOAD EMBEDDINGS
#====================================================================

print("\nLoading embeddings...")

train_data = torch.load("train_embeddings.pt")
cal_data = torch.load("cal_embeddings.pt")
test_data = torch.load("test_embeddings.pt")

train_embeddings = train_data["embeddings"]  # (60000, 128)
train_labels = train_data["labels"]  # (60000,)
cal_embeddings = cal_data["embeddings"]  # (1000, 128)
cal_labels = cal_data["labels"]  # (1000,)
test_embeddings = test_data["embeddings"]  # (9000, 128)
test_labels = test_data["labels"]  # (9000,)

print(f"Training embeddings:    {train_embeddings.shape}")
print(f"Calibration embeddings: {cal_embeddings.shape}")
print(f"Test embeddings:        {test_embeddings.shape}")

#====================================================================
## 2. ORGANISE EMBEDDINGS BY CLASS
#====================================================================
print("\nOrganising training embeddings by class...")

NUM_CLASSES = 10
class_embeddings = {}
# creates empty python dictionary - stores key-value pairs (look up key, get value)
# key = digit (0,..,9) , value = tensor of all training embeddings associated with that digit
# eg. class_embeddings[7] gets all ~6,000 training embeddings of 7s
for digit in range(NUM_CLASSES):
    mask = (train_labels == digit)
    class_embeddings[digit] = train_embeddings[mask]  # (N_digit, 128)
    print(f"  Digit {digit}: {len(class_embeddings[digit]):,} training examples available")

#====================================================================
## 3. NEAREST NEIGHBOUR DISTANCE FUCTION
#====================================================================
# query embedding = embedding of the test image
# input batch of query embeddings, compute the distance to the nearest training example
# from each class. shape: (N , 128)
# output distances. shape: (N, 10) (one per class)

# more computationally expensive than the centroid distance method.
# on fin. docs. with BERT could be a more important tradeoff

def nearest_neighbour_distances(query_embeddings, class_embeddings, k, metric):
    N = len(query_embeddings)
    distances = torch.zeros(N, NUM_CLASSES)

    for digit in range(NUM_CLASSES):
        class_embs = class_embeddings[digit]  # (N_digit, 128)

        if metric == "euclidean":
            # pairwise distance between every query point and every
            # training point of this class. shape: (N, N_digit)
            pairwise_dists = torch.cdist(query_embeddings, class_embs, p=2)
        elif metric == "cosine":
            q_norm = query_embeddings / query_embeddings.norm(dim=1, keepdim=True).clamp(min=1e-8)
            c_norm = class_embs / class_embs.norm(dim=1, keepdim=True).clamp(min=1e-8)
            pairwise_dists = 1 - (q_norm @ c_norm.T)

        if k == 1:
            # simplest case — just take the minimum distance for each query point
            distances[:, digit] = pairwise_dists.min(dim=1).values
        else:
            # average distance to the k nearest training points
            topk_dists = pairwise_dists.topk(k, dim=1, largest=False).values
            distances[:, digit] = topk_dists.mean(dim=1)

    return distances  # (N, 10)

# =============================================================================
# 4. COMPUTE NONCONFORMITY SCORES ON CALIBRATION SET
# =============================================================================
# find distance to nearest true class embedding for each calibration example.
# searches through all 60,000 training embeddings.

print("\nComputing calibration nonconformity scores...")
print("\nTakes a few minutes...")
cal_distances_all_classes = nearest_neighbour_distances(
    cal_embeddings, class_embeddings, K, DISTANCE_METRIC
)  # (1000, 10)

# distance to TRUE class for each calibration example
cal_true_distances = cal_distances_all_classes.gather(
    dim=1,
    index=cal_labels.unsqueeze(1)
).squeeze(1)  # (1000,)

cal_scores = cal_true_distances

print(f"\nCalibration nonconformity scores:")
print(f"  Mean:   {cal_scores.mean():.4f}")
print(f"  Median: {cal_scores.median():.4f}")
print(f"  Max:    {cal_scores.max():.4f}")
print(f"  Min:    {cal_scores.min():.4f}")

# =============================================================================
# 5. COMPUTE CONFORMAL THRESHOLD QHAT
# =============================================================================

n = len(cal_scores)
level = np.ceil((n + 1) * (1 - ALPHA)) / n
level = min(level, 1.0)
qhat = torch.quantile(cal_scores, level)

print(f"\nConformal threshold (qhat): {qhat:.4f}")
print(f"Interpretation: any class with a training example within distance {qhat:.4f} of the test embedding gets included")

# =============================================================================
# 6. BUILD PREDICTION SETS ON TEST SET
# =============================================================================
print("\nBuilding prediction sets...")
print("(Searching through all 60,000 training embeddings for 9,000 test points)")

test_distances_all_classes = nearest_neighbour_distances(
    test_embeddings, class_embeddings, K, DISTANCE_METRIC
)  # (9000, 10)

prediction_sets = test_distances_all_classes <= qhat  # (9000, 10) boolean

print(f"Prediction sets built for {len(prediction_sets):,} test examples")

# =============================================================================
# 7. EVALUATE COVERAGE AND EFFICIENCY
# =============================================================================
correct_coverage = prediction_sets.gather(
    dim=1,
    index=test_labels.unsqueeze(1)
).squeeze(1)

empirical_coverage = correct_coverage.float().mean().item()
set_sizes = prediction_sets.sum(dim=1).float()
avg_set_size = set_sizes.mean().item()

print(f"\n{'=' * 55}")
print(f"RESULTS — NEAREST NEIGHBOUR CONFORMAL INFERENCE (k={K})")
print(f"{'=' * 55}")
print(f"Distance metric:    {DISTANCE_METRIC}")
print(f"Target coverage:    {(1 - ALPHA) * 100:.0f}%")
print(f"Empirical coverage: {empirical_coverage * 100:.2f}%")
print(f"Average set size:   {avg_set_size:.3f}")
print(f"{'=' * 55}")

if empirical_coverage >= (1 - ALPHA):
    print("✓ Coverage guarantee holds")
else:
    print("✗ Coverage guarantee does not hold")

# =============================================================================
# 8. SET SIZE DISTRIBUTION
# =============================================================================

print(f"\nSet size distribution:")
for size in range(0, 11):
    count = (set_sizes == size).sum().item()
    percent = count / len(set_sizes) * 100
    bar = "█" * int(percent / 2)
    print(f"  Size {size:2d}: {count:5,} ({percent:5.1f}%) {bar}")

# should produce less empty sets than centroid scoring

# =============================================================================
# 9. EXAMPLE PREDICTION SETS
# =============================================================================

print(f"\nExample prediction sets (first 10 test images):")
print(f"{'Example':<10} {'True':<8} {'Prediction set':<22} {'Size':<6} {'Covered'}")
print("-" * 58)
for i in range(10):
    true_label = test_labels[i].item()
    pred_set = prediction_sets[i].nonzero(as_tuple=True)[0].tolist()
    size = len(pred_set)
    covered = true_label in pred_set
    print(f"{i:<10} {true_label:<8} {str(pred_set):<22} {size:<6} {'✓' if covered else '✗'}")






