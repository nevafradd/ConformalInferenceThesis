#===========================================================================
# Conformal Inference pipeline for MNIST
#============================================================================
#
#
# Inputs:
# mnist_cnn_scores : softmax probabilites and true labels (saved after cnn
# calibration split: chunk of 10,000 to compute threshold
# remaining test examples

# What happens?
# Splits saved scores into calibration and test
# computes non-conformity for every calibration example
# sorts scores and finds threshold at 95%
# for each test image, builds prediction set

# Outputs:
# prediction set for each image (rather than single label)
# empirical coverage (>= 95%)
# average prediction set size -> more uncertain, larger prediction sets

# So we have a prediction set with a guaranteed coverage

# =============================================================================

import torch
import torch.nn.functional as F
import numpy as np

# =============================================================================
# 0. CONFIGURATION
# =============================================================================

# CODING CHOICES!!
ALPHA = 0.05 # coverage guarantee 95%
N_CALIBRATION = 1000 # calibration set size

# =============================================================================
# 1. LOAD SAVED SCORES
# =============================================================================
# Load softmax probabilities and true labels saved by mnist_cnn.py.

print("Loading saved softmax scores...")
checkpoint = torch.load("mnist_cnn_scores.pt")
probs      = checkpoint["probs"]    # (10000, 10) 10 probs per image
labels     = checkpoint["labels"]   # (10000,) # 1 label per image

print(f"Loaded {len(probs):,} examples")
print(f"Probs shape:  {probs.shape}")
print(f"Labels shape: {labels.shape}")

# =============================================================================
# 2. SPLIT INTO CALIBRATION AND TEST SETS
# =============================================================================
#
# CODING CHOICE: CALIBRATION SET SIZE, RANDOM SHUFFLE
torch.manual_seed(42) # makes split reproducible
indices   = torch.randperm(len(probs))   # random permutation of 0..9999
probs     = probs[indices]
labels    = labels[indices]

cal_probs    = probs[:N_CALIBRATION]     # (1000, 10)
cal_labels   = labels[:N_CALIBRATION]   # (1000,)
test_probs   = probs[N_CALIBRATION:]    # (9000, 10)
test_labels  = labels[N_CALIBRATION:]  # (9000,)

print(f"\nCalibration set: {len(cal_probs):,} examples")
print(f"Test set:        {len(test_probs):,} examples")

# should use calibration data held out from CNN training, not part of test data?

# =============================================================================
# 3. COMPUTE NONCONFORMITY SCORES ON CALIBRATION SET
# =============================================================================
# through all calibration, find probability of true class
#
# torch.gather pulls out needed probabilities across whole batch simultaneously

cal_true_class_probs = cal_probs.gather(
    dim=1, # operate along class - 10 values
    index=cal_labels.unsqueeze(1)   # shape (1000,) → (1000, 1) for gather
).squeeze(1)                        # back to (1000,)

cal_scores = 1 - cal_true_class_probs   # nonconformity scores, shape (1000,)
# outputs a list of 1000 numbers (non-conf scores)

print(f"\nCalibration nonconformity scores:")
print(f"  Mean:   {cal_scores.mean():.4f}")
print(f"  Median: {cal_scores.median():.4f}")
print(f"  Max:    {cal_scores.max():.4f}")
print(f"  Min:    {cal_scores.min():.4f}")

# =============================================================================
# 4. FIND THE CONFORMAL THRESHOLD (QHAT)
# =============================================================================
# Order the calibration scores, and find value at 0.95 quantile (95% of values
# lie below it)
# this value is the conformal threshold. any class with a non-conformity score
# below qhat is included in the prediction set.
#
# Formula used to find conformal threshold is proven in literature - guarantees
# exact finite-sample coverage # (Vovk et al., 2005; Angelopoulos & Bates, 2021).

# The exact formula is: ceil((n+1)(1-alpha)) / n - n is the calibration set size.
#
n     = len(cal_scores)
level = np.ceil((n + 1) * (1 - ALPHA)) / n
level = min(level, 1.0)   # makes sure quantile doesn't go over 1

qhat  = torch.quantile(cal_scores, level)

print(f"\nConformal threshold (qhat): {qhat:.4f}")
print(f"Coverage target: {(1 - ALPHA) * 100:.0f}%")
print(f"Interpretation: any class with nonconformity score <= {qhat:.4f}")
print(f"                gets included in the prediction set")

# =============================================================================
# 5. BUILD PREDICTION SETS ON TEST SET
# =============================================================================
# For each test image, compute nonconformity scores for ALL 10 classes
# and include any class whose score falls at or below qhat.
#
# nonconformity score for class c = 1 - softmax_prob[c]
# include class c if: 1 - softmax_prob[c] <= qhat
# equivalently:       softmax_prob[c] >= 1 - qhat
#

# fixed - guarantees no empty sets
# sort classes by probability descending, include until cumulative prob >= 1-alpha
sorted_probs, sorted_indices = test_probs.sort(dim=1, descending=True)
cumulative_probs = sorted_probs.cumsum(dim=1)
# include classes until cumulative probability exceeds the threshold
include = cumulative_probs - sorted_probs < (1 - ALPHA)
# scatter back to original class ordering
prediction_sets = torch.zeros_like(test_probs, dtype=torch.bool)
prediction_sets.scatter_(1, sorted_indices, include)


# output = True in position (i, c) means class c is in the prediction set for
# image i

# check we have 9,000 prediction sets - one per test example
print(f"\nPrediction sets built for {len(prediction_sets):,} test examples")

# =============================================================================
# 6. EVALUATE COVERAGE AND EFFICIENCY
# =============================================================================
# Coverage: what fraction of test examples contain true label in their set?
# should be >= 95%
#
# Efficiency: average prediction set size.
# Smaller is better — larger means more uncertainty.

# Coverage: for each test example, check if true label is in prediction set
correct_coverage = prediction_sets.gather(
    dim=1,
    # unsqueeze turns (9000,) to (9000,1) so gather can use it
    index=test_labels.unsqueeze(1)
).squeeze(1)
# outputs a flat list of True/ False values. True if the true class is included

empirical_coverage = correct_coverage.float().mean().item()
# converts T/F to 1/0 , averages all values (giving proportion of test images
# with true label contained) , convert to printable float.

# Efficiency: average number of classes in each prediction set
set_sizes          = prediction_sets.sum(dim=1).float()
avg_set_size       = set_sizes.mean().item()

print(f"\n{'='*50}")
print(f"RESULTS")
print(f"{'='*50}")
print(f"Target coverage:    {(1 - ALPHA) * 100:.0f}%")
print(f"Empirical coverage: {empirical_coverage * 100:.2f}%")
print(f"Average set size:   {avg_set_size:.3f}")
print(f"{'='*50}")

if empirical_coverage >= (1 - ALPHA):
    print("✓ Coverage guarantee holds on this test set")
else:
    print("✗ Coverage guarantee does not hold — check calibration split")

# =============================================================================
# 7. SET SIZE DISTRIBUTION
# =============================================================================
# How often does the model produce sets of size 1, 2, 3 etc?
#For MNIST, should be mostly size 1
# prints histogram for set sizes 1 to 10. set size 0 is very unlikely due to qhat
# formula correction
print(f"\nSet size distribution:")
for size in range(1, 11):
    count   = (set_sizes == size).sum().item()
    percent = count / len(set_sizes) * 100
    bar     = "█" * int(percent / 2)
    print(f"  Size {size:2d}: {count:5,} ({percent:5.1f}%) {bar}")

# =============================================================================
# 8. SHOW EXAMPLE PREDICTION SETS
# =============================================================================
#see what prediction sets look like in practice.
#

print(f"\nExample prediction sets (first 10 test images):")
print(f"{'Example':<10} {'True label':<12} {'Prediction set':<20} {'Size':<6} {'Covered'}")
print("-" * 60)

for i in range(10):
    true_label  = test_labels[i].item()
    pred_set    = prediction_sets[i].nonzero(as_tuple=True)[0].tolist()
    size        = len(pred_set)
    covered     = true_label in pred_set
    print(f"{i:<10} {true_label:<12} {str(pred_set):<20} {size:<6} {'✓' if covered else '✗'}")


# =============================================================================
# 1. Empirical coverage should be >= 95%. If it's below, something is wrong
#    with the calibration split or the threshold computation.
#
# 2. Average set size should be close to 1.0 for MNIST. If it's consistently
#    2 or 3, your CNN may not be confident enough — consider training longer.
#
# 3. The set size distribution should be heavily concentrated at size 1,
#    with occasional size 2 on ambiguous digits (e.g. 4 vs 9, 3 vs 8).
#
#========================================================================