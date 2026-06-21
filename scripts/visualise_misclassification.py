### in the relative distance embedding CI we had 524 empty sets, which indicated 524 embeddings that belonged to the wrong cluster
# this script will take those embeddings and look at the images, and see what the mistakes are like
# look at a grid of sample images
# look at a UMAP plot with indicated mistakes

import torch
import numpy as np
import matplotlib.pyplot as plt
import csv
from torchvision import datasets, transforms
import umap

# CONFIGURATION = MUST MATCH PREV. SCRIPTS

N_CALIBRATION   = 1000
ALPHA           = 0.05
EPSILON         = 1e-8
N_GRID_EXAMPLES = 20      # *** CHOICE: how many to show in the image grid ***
RANDOM_SEED     = 42

print("Loading embeddings...")
train_data = torch.load("train_embeddings.pt")
cal_data = torch.load("cal_embeddings.pt")
test_data = torch.load("test_embeddings.pt")

train_embeddings = train_data["embeddings"]
train_labels = train_data["labels"]
cal_embeddings = cal_data["embeddings"]
cal_labels = cal_data["labels"]
test_embeddings = test_data["embeddings"]
test_labels = test_data["labels"]

## RECOMPUTE CENTROIDS, SCORES, QHAT

NUM_CLASSES = 10
EMBEDDING_DIM = 128
centroids = torch.zeros(NUM_CLASSES, EMBEDDING_DIM)

print("Computing class centroids...")
for digit in range(NUM_CLASSES):
    mask = (train_labels == digit)
    centroids[digit] = train_embeddings[mask].mean(dim=0)


def compute_distances(embeddings, centroids):
    return torch.cdist(embeddings, centroids, p=2)


def relative_score_for_true_class(embeddings, labels, centroids):
    all_distances = compute_distances(embeddings, centroids)
    true_class_dist = all_distances.gather(dim=1, index=labels.unsqueeze(1)).squeeze(1)
    masked = all_distances.clone()
    masked.scatter_(dim=1, index=labels.unsqueeze(1), value=float('inf'))
    nearest_other = masked.min(dim=1).values
    return true_class_dist / (nearest_other + EPSILON)


def relative_scores_for_all_classes(embeddings, centroids):
    all_distances = compute_distances(embeddings, centroids)
    N = len(embeddings)
    scores = torch.zeros(N, NUM_CLASSES)
    for c in range(NUM_CLASSES):
        dist_to_c = all_distances[:, c]
        masked = all_distances.clone()
        masked[:, c] = float('inf')
        nearest_other = masked.min(dim=1).values
        scores[:, c] = dist_to_c / (nearest_other + EPSILON)
    return scores


print("Computing qhat from calibration set...")
cal_scores = relative_score_for_true_class(cal_embeddings, cal_labels, centroids)
n = len(cal_scores)
level = np.ceil((n + 1) * (1 - ALPHA)) / n
level = min(level, 1.0)
qhat = torch.quantile(cal_scores, level)
print(f"qhat = {qhat:.4f}")

## BUILD PREDICTION SETS , FIND MISPLACED EXAMPLES

print("\nBuilding prediction sets (no safety-net patch — reverted as discussed)...")

test_scores_all = relative_scores_for_all_classes(test_embeddings, centroids)
prediction_sets = test_scores_all <= qhat  # (9000, 10) boolean

set_sizes = prediction_sets.sum(dim=1)
empty_mask = (set_sizes == 0)  # boolean, True where set is empty
empty_idx = empty_mask.nonzero(as_tuple=True)[0]  # positions in test_embeddings order

print(f"Total empty-set examples found: {len(empty_idx)}")

# also find, for each empty-set example, which class it WAS closest to
# (i.e. which wrong class "stole" the spot)
test_distances = compute_distances(test_embeddings, centroids)
nearest_class_all = test_distances.argmin(dim=1)  # (9000,) — model's best guess

### MAP EMBEDDING POSITIONS BACK TO ORIGINAL MNIST TEST IMAGES

print("Mapping back to original MNIST test set images...")

torch.manual_seed(RANDOM_SEED)
full_test_dataset = datasets.MNIST(
    root="./data", train=False, download=True,
    transform=transforms.ToTensor()
)
full_indices = torch.randperm(len(full_test_dataset))
test_idx_map = full_indices[N_CALIBRATION:]  # embedding position -> original MNIST index

### SAVE FULL LIST OF ALL EMPTY-SET EXAMPLES TO CSV

print(f"\nSaving full list of {len(empty_idx)} empty-set examples to CSV...")

with open("empty_set_examples.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "embedding_position", "original_mnist_index", "true_label",
        "nearest_wrong_class", "true_class_score", "distance_to_true_centroid",
        "distance_to_nearest_class"
    ])

    true_scores_test = relative_score_for_true_class(test_embeddings, test_labels, centroids)

    for idx in empty_idx.tolist():
        original_idx = test_idx_map[idx].item()
        true_label = test_labels[idx].item()
        nearest_wrong = nearest_class_all[idx].item()
        score = true_scores_test[idx].item()
        dist_to_true = test_distances[idx, true_label].item()
        dist_to_nearest = test_distances[idx, nearest_wrong].item()

        writer.writerow([
            idx, original_idx, true_label, nearest_wrong,
            f"{score:.4f}", f"{dist_to_true:.4f}", f"{dist_to_nearest:.4f}"
        ])

print("Saved: empty_set_examples.csv")
print("Open this in Excel or pandas to inspect all 524 cases in detail.")

## SUMMARY STATISTICS — WHICH DIGITS ARE MOST OFTEN MISPLACED?

print("\nWhich true digits appear most often among empty-set examples?")
empty_true_labels = test_labels[empty_idx]
for digit in range(10):
    count = (empty_true_labels == digit).sum().item()
    total_of_digit = (test_labels == digit).sum().item()
    pct = count / total_of_digit * 100 if total_of_digit > 0 else 0
    print(f"  True digit {digit}: {count:3d} empty sets out of {total_of_digit} "
          f"({pct:.1f}% of all test examples of this digit)")

print("\nWhich WRONG class most often 'steals' the spot?")
empty_wrong_labels = nearest_class_all[empty_idx]
for digit in range(10):
    count = (empty_wrong_labels == digit).sum().item()
    print(f"  Wrongly nearest to digit {digit}: {count} times")

# =============================================================================
# 7. VISUALISATION 1 — GRID OF SAMPLE MISPLACED IMAGES
# =============================================================================

print(f"\nGenerating image grid of {N_GRID_EXAMPLES} misplaced examples...")

n_show = min(N_GRID_EXAMPLES, len(empty_idx))
n_cols = 5
n_rows = int(np.ceil(n_show / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

for plot_i in range(n_show):
    idx = empty_idx[plot_i].item()
    original_idx = test_idx_map[idx].item()
    image, _ = full_test_dataset[original_idx]
    image = image.squeeze().numpy()

    true_label = test_labels[idx].item()
    wrong_class = nearest_class_all[idx].item()
    score = true_scores_test[idx].item()

    ax = axes[plot_i]
    ax.imshow(image, cmap="gray")
    ax.set_title(
        f"True: {true_label}  Nearest: {wrong_class}\nScore: {score:.3f}",
        fontsize=9, color="darkred"
    )
    ax.axis("off")

# hide any unused subplot axes
for plot_i in range(n_show, len(axes)):
    axes[plot_i].axis("off")

fig.suptitle(
    f"Empty-Set Examples — Test Images Whose True Class Was Excluded\n"
    f"({len(empty_idx)} total, showing first {n_show})",
    fontsize=13, fontweight="bold"
)

plt.tight_layout()
plt.savefig("empty_set_images_grid.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: empty_set_images_grid.png")

# =============================================================================
# 8. VISUALISATION 2 — WHERE DO THESE POINTS SIT IN EMBEDDING SPACE?
# =============================================================================
# Project a sample of normal test points plus ALL empty-set points into 2D
# using UMAP, and highlight the empty-set points in red. This shows whether
# they cluster near specific decision boundaries or are scattered randomly.

print("\nGenerating embedding space visualisation (UMAP)...")

# build a combined set: a random sample of "normal" points + all empty-set points
N_BACKGROUND = 3000
non_empty_idx = (~empty_mask).nonzero(as_tuple=True)[0]
background_sample = non_empty_idx[torch.randperm(len(non_empty_idx))[:N_BACKGROUND]]

combined_idx = torch.cat([background_sample, empty_idx])
combined_embeddings = test_embeddings[combined_idx].numpy()
combined_is_empty = torch.cat([
    torch.zeros(len(background_sample), dtype=torch.bool),
    torch.ones(len(empty_idx), dtype=torch.bool)
]).numpy()
combined_labels = test_labels[combined_idx].numpy()

reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                    metric="euclidean", random_state=RANDOM_SEED)
coords_2d = reducer.fit_transform(combined_embeddings)
centroids_2d = reducer.transform(centroids.numpy())

fig, ax = plt.subplots(figsize=(12, 10))

# plot background points faded, coloured by true class
colours = plt.cm.tab10(np.linspace(0, 1, 10))
for digit in range(10):
    mask = (combined_labels == digit) & (~combined_is_empty)
    ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1],
               c=[colours[digit]], alpha=0.15, s=8, label=f"Digit {digit}")

# overlay empty-set points prominently in red
ax.scatter(coords_2d[combined_is_empty, 0], coords_2d[combined_is_empty, 1],
           c="red", s=35, marker="x", linewidths=1.5,
           label=f"Empty-set examples (n={len(empty_idx)})", zorder=5)

# overlay centroids
ax.scatter(centroids_2d[:, 0], centroids_2d[:, 1],
           c=colours, s=250, marker="*", edgecolors="black",
           linewidths=1, zorder=6, label="_nolegend_")
for digit in range(10):
    ax.annotate(str(digit), (centroids_2d[digit, 0], centroids_2d[digit, 1]),
                fontsize=13, fontweight="bold", ha="center", va="bottom",
                xytext=(0, 8), textcoords="offset points")

ax.set_title(
    "Where Empty-Set Examples Sit in Embedding Space\n"
    "Red X = true class excluded from prediction set. Stars = class centroids.",
    fontsize=13, fontweight="bold", pad=15
)
ax.set_xlabel("UMAP Component 1")
ax.set_ylabel("UMAP Component 2")
ax.legend(loc="upper right", markerscale=1.5, fontsize=8, framealpha=0.9, ncol=2)
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("empty_set_in_embedding_space.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: empty_set_in_embedding_space.png")

