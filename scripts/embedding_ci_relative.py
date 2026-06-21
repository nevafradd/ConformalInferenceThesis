# most sophisticated of embeddings based conformal inference.
# never produces empty sets

# looks at how close a point is to it's true centroid, compared to the nearest other class centroid
# directly measures proximity to decision boundaries

# relative distance score = distance to true centroid/ distance to next nearest centroid

import torch
import numpy as np

# CONFIGURATION

ALPHA = 0.05
DISTANCE_METRIC = "euclidean"

print(f"distance metric: {DISTANCE_METRIC}")
print(f"coverage target: {(1-ALPHA)*100:.0f}%")

# LOAD EMBEDDINGS
##====================================================

print("\nLoading embeddings...")

train_data = torch.load("train_embeddings.pt")
cal_data = torch.load("cal_embeddings.pt")
test_data = torch.load("test_embeddings.pt")

train_embeddings = train_data["embeddings"]
train_labels = train_data["labels"]
cal_embeddings = cal_data["embeddings"]
cal_labels = cal_data["labels"]
test_embeddings = test_data["embeddings"]
test_labels = test_data["labels"]

print(f"Training embeddings:    {train_embeddings.shape}")
print(f"Calibration embeddings: {cal_embeddings.shape}")
print(f"Test embeddings:        {test_embeddings.shape}")

# =============================================================================
# 2. COMPUTE CLASS CENTROIDS
# =============================================================================
#done before in embedding_ci_centroid.py
NUM_CLASSES = 10
EMBEDDING_DIM = 128
centroids = torch.zeros(NUM_CLASSES, EMBEDDING_DIM)

print("\nComputing class centroids...")
for digit in range(NUM_CLASSES):
    mask = (train_labels == digit)
    centroids[digit] = train_embeddings[mask].mean(dim=0)
    print(f"  Digit {digit}: centroid computed from {mask.sum():,} embeddings")


# =============================================================================
# 3. DISTANCE FUNCTION
# =============================================================================
# computes distance from a batch of embeddings to all 10 centroids.

def compute_distances(embeddings, centroids, metric):
    if metric == "euclidean":
        return torch.cdist(embeddings, centroids, p=2)  # (N, 10)
    elif metric == "cosine":
        emb_norm = embeddings / embeddings.norm(dim=1, keepdim=True).clamp(min=1e-8)
        cen_norm = centroids / centroids.norm(dim=1, keepdim=True).clamp(min=1e-8)
        return 1 - (emb_norm @ cen_norm.T)

# =============================================================================
# 4. RELATIVE DISTANCE SCORE FUNCTION
# =============================================================================
# new section:score = distance to true centroid/ distance to next nearest centroid
# on calibration: compute score using known TRUE label
# on test: compute score for every possible class, treat each class as if it's TRUE
#         in turn to decide whether to keep it in the prediction set

EPSILON = 1e-8
# This is to make sure we're not dividing by 0 - if a point sits exactly on a centroid

def relative_score_for_true_class(embeddings, labels, centroids, metric):
    """
    For each embedding, compute distance to its TRUE class centroid divided
    by distance to the nearest OTHER class centroid.
    Returns shape (N,) — one score per embedding.
    """
    all_distances = compute_distances(embeddings, centroids, metric)  # (N, 10)
    N = len(embeddings)

    # distance to true class centroid
    true_class_dist = all_distances.gather(
        dim=1, index=labels.unsqueeze(1)
    ).squeeze(1)  # (N,)

    # to find the distance to the nearest other class - set true class column to infinity (masks)
    # and then take the min across the row
    masked_distances = all_distances.clone()
    masked_distances.scatter_(
        dim=1, index=labels.unsqueeze(1), value=float('inf')
    )
    nearest_other_dist = masked_distances.min(dim=1).values  # (N,)

    scores = true_class_dist / (nearest_other_dist + EPSILON)
    return scores


def relative_scores_for_all_classes(embeddings, centroids, metric):
    """
    For each embedding, compute the relative distance score AS IF each of
    the 10 classes were the true class. Used at test time when we don't
    know the true label and need to decide, for every candidate class,
    whether to include it in the prediction set.
    Returns shape (N, 10).
    """
    all_distances = compute_distances(embeddings, centroids, metric)  # (N, 10)
    N = len(embeddings)
    scores = torch.zeros(N, NUM_CLASSES)

    for candidate_class in range(NUM_CLASSES):
        dist_to_candidate = all_distances[:, candidate_class]  # (N,)

        # nearest OTHER class distance — mask out the candidate column
        masked = all_distances.clone()
        masked[:, candidate_class] = float('inf')
        nearest_other = masked.min(dim=1).values  # (N,)

        scores[:, candidate_class] = dist_to_candidate / (nearest_other + EPSILON)

    return scores

# =============================================================================
# 5. COMPUTE NONCONFORMITY SCORES ON CALIBRATION SET
# =============================================================================
# use true label here (relative_score_for_true_class)

print("\nComputing calibration nonconformity scores...")

cal_scores = relative_score_for_true_class(
    cal_embeddings, cal_labels, centroids, DISTANCE_METRIC
)  # (1000,)

print(f"Calibration nonconformity scores:")
print(f"  Mean:   {cal_scores.mean():.4f}")
print(f"  Median: {cal_scores.median():.4f}")
print(f"  Max:    {cal_scores.max():.4f}")
print(f"  Min:    {cal_scores.min():.4f}")
print(f"  (Scores near 0 = confidently correct class.")
print(f"   Scores near 1 = sitting on a decision boundary.")
print(f"   Scores above 1 = closer to a different class than the true one.)")

print(f"Fraction of cal scores below 1.0: {(cal_scores < 1.0).float().mean():.4f}")
print(f"Fraction of cal scores below 0.5: {(cal_scores < 0.5).float().mean():.4f}")
sorted_scores = cal_scores.sort().values
print(f"Score at 90th percentile index: {sorted_scores[900]:.4f}")
print(f"Score at 95th percentile index: {sorted_scores[950]:.4f}")
print(f"Score at 99th percentile index: {sorted_scores[990]:.4f}")

# =============================================================================
# 6. COMPUTE QHAT
# =============================================================================

n = len(cal_scores)
level = np.ceil((n + 1) * (1 - ALPHA)) / n
level = min(level, 1.0)
qhat = torch.quantile(cal_scores, level)

print(f"\nConformal threshold (qhat): {qhat:.4f}")
print(f"Interpretation: any class with relative score <= {qhat:.4f} gets included in the prediction set")

##===========================================================================
## 7. BUILD PREDICTION SETS ON TEST SET
##===========================================================================
# treat every class as a potential true class

print("\nBuilding prediction sets...")

test_scores_all_classes = relative_scores_for_all_classes(
    test_embeddings, centroids, DISTANCE_METRIC
)  # (9000, 10)

prediction_sets = test_scores_all_classes <= qhat  # (9000, 10) boolean

print(f"Prediction sets built for {len(prediction_sets):,} test examples")


# =============================================================================
# 8. EVALUATE COVERAGE AND EFFICIENCY
# =============================================================================
correct_coverage = prediction_sets.gather(
    dim=1, index=test_labels.unsqueeze(1)
).squeeze(1)

empirical_coverage = correct_coverage.float().mean().item()
set_sizes = prediction_sets.sum(dim=1).float()
avg_set_size = set_sizes.mean().item()

print(f"\n{'=' * 55}")
print(f"RESULTS — RELATIVE DISTANCE CONFORMAL INFERENCE")
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

empty_sets = (set_sizes == 0).sum().item()
print(f"\nEmpty sets: {empty_sets} (should be rare or zero by construction)")

# =============================================================================
# 9. SET SIZE DISTRIBUTION
# =============================================================================

print(f"\nSet size distribution:")
for size in range(0, 11):
    count = (set_sizes == size).sum().item()
    percent = count / len(set_sizes) * 100
    bar = "█" * int(percent / 2)
    print(f"  Size {size:2d}: {count:5,} ({percent:5.1f}%) {bar}")

# =============================================================================
# 10. EXAMPLE PREDICTION SETS — INCLUDING BOUNDARY CASES
# =============================================================================
# find the most ambiguous embeddings what have high relative scores for their true class
# check images to see if they are actually hard to classify

print(f"\nExample prediction sets (first 10 test images):")
print(f"{'Example':<10} {'True':<8} {'Prediction set':<22} {'Size':<6} {'Covered'}")
print("-" * 58)
for i in range(10):
    true_label = test_labels[i].item()
    pred_set = prediction_sets[i].nonzero(as_tuple=True)[0].tolist()
    size = len(pred_set)
    covered = true_label in pred_set
    print(f"{i:<10} {true_label:<8} {str(pred_set):<22} {size:<6} {'✓' if covered else '✗'}")

print(f"\nMost ambiguous test examples (highest relative score for true class):")
print(f"{'Example':<10} {'True':<8} {'Score':<10} {'Prediction set':<22} {'Size'}")
print("-" * 60)

true_scores_test = relative_score_for_true_class(
    test_embeddings, test_labels, centroids, DISTANCE_METRIC
)
most_ambiguous_idx = true_scores_test.argsort(descending=True)[:10]

for idx in most_ambiguous_idx:
    idx = idx.item()
    true_label = test_labels[idx].item()
    score = true_scores_test[idx].item()
    pred_set = prediction_sets[idx].nonzero(as_tuple=True)[0].tolist()
    size = len(pred_set)
    print(f"{idx:<10} {true_label:<8} {score:<10.4f} {str(pred_set):<22} {size}")


# distribution of the MINIMUM relative score across the 10 candidates,
# for each test point — this tells us what score the model's best guess
# typically gets
min_test_scores = test_scores_all_classes.min(dim=1).values
print(f"Min test score - mean: {min_test_scores.mean():.4f}")
print(f"Min test score - median: {min_test_scores.median():.4f}")
print(f"Fraction of min test scores below qhat: {(min_test_scores < qhat).float().mean():.4f}")




