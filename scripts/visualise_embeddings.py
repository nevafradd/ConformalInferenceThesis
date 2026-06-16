# GOAL: to visualise the 128-dimensional embeddings from our CNN on MNIST
# look at how digit classes cluster together, where boundaries are, which examples
# are ambiguous

# outputs:
# 1. UMAP projection coloured by class
#2. t-SNE projection coloured by class
# 3. UMAP with overlaid centroids
# 4. heatmap of pairwise centroid distributions

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
import umap

## CONFIGURATION
# Choose how many training examples to visualise
N_SAMPLES = 10000
RANDOM_SEED = 42
DIGIT_NAMES = [str(i) for i in range(10)]
#tab10 colour palette - designed for 10 categories
COLOURS = plt.cm.tab10(np.linspace(0, 1, 10))

print(f"Visualising {N_SAMPLES} embeddings")
print(f"Random Seed: {RANDOM_SEED}")

# =============================================================================
# 1. LOAD EMBEDDINGS
# =============================================================================

print("\nLoading training embeddings...")
train_data       = torch.load("train_embeddings.pt")
train_embeddings = train_data["embeddings"]   # (60000, 128)
train_labels     = train_data["labels"]       # (60000,)

print(f"Loaded {len(train_embeddings):,} embeddings of dimension {train_embeddings.shape[1]}")

# =============================================================================
# 2. SUBSAMPLE FOR VISUALISATION
# =============================================================================
#randomly select 10,000 samples to plot - stratified sampling method to make sure
#each digit is equally represented.

print(f"\nSubsampling training embeddings...")

samples_per_class = N_SAMPLES // 10
sample_indices = []

for i in range(10):
    class_indices = (train_labels == i).nonzero(as_tuple=True)[0]
    perm = torch.randperm(len(class_indices))[:samples_per_class]
    sample_indices.append(class_indices[perm])

sample_indices = torch.cat(sample_indices)
embeddings_sample = train_embeddings[sample_indices].numpy()
labels_sample = train_labels[sample_indices].numpy()

print(f"Sampled {len(embeddings_sample):,} embeddings ({samples_per_class} per class)")

# =============================================================================
# 3. COMPUTE CLASS CENTROIDS
# =============================================================================

# the same computation as we did in the CI code - needed for the overlay plot

print("\nComputing class centroids...")
centroids = np.zeros((10, 128))
for digit in range(10):
    mask = (train_labels.numpy() == digit)
    centroids[digit] = train_embeddings.numpy()[mask].mean(axis=0)

# =============================================================================
# 4. UMAP PROJECTION
# =============================================================================
# uniform manifold approximation and projection
# preserves local and global structure of the 128-dimensional space

# Key parameters:
#### n_neighbors — how many neighbours to consider when learning local structure
#lower = more local detail, higher = more global structure
# 15 is the standard default
#### min_dist — how tightly to pack points in 2D
#  lower = tighter clusters, higher = more spread out
# 0.1 is the standard default
#### metric — distance metric to use in the original space
# euclidean, same as before

print("\nRunning UMAP (this takes ~30 seconds)...")
reducer = umap.UMAP(
    n_components=2,
    n_neighbors=15,  # *** CHOICE 3: local vs global structure ***
    min_dist=0.1,  # *** CHOICE 4: cluster tightness ***
    metric="euclidean",
    random_state=RANDOM_SEED
)

umap_2d = reducer.fit_transform(embeddings_sample)  # (N_SAMPLES, 2)
print(f"UMAP complete. Output shape: {umap_2d.shape}")

# also project centroids into the same 2D space
centroids_2d_umap = reducer.transform(centroids)  # (10, 2)

# =============================================================================
# 5. t-SNE PROJECTION
# =============================================================================
# t-distributed stochastic neighbour embedding - preserves local structure
# distances between clusters not good
# reliable structure within and immediately around clusters is reliable

# perplexity — roughly the number of neighbours each point considers.
# typical range 5-50. 30 is a standard default.

print("Running t-SNE (this takes ~2 minutes)...")
tsne = TSNE(
    n_components=2,
    perplexity=30,  # *** CHOICE 5: neighbourhood size ***
    random_state=RANDOM_SEED,
    max_iter=1000  # more iterations = more stable layout
)

tsne_2d = tsne.fit_transform(embeddings_sample)  # (N_SAMPLES, 2)
print(f"t-SNE complete. Output shape: {tsne_2d.shape}")


# =============================================================================
# 6. PLOTTING HELPER
# =============================================================================

def plot_embeddings(coords_2d, labels, title, filename,
                    centroids_2d=None, highlight_indices=None):
    fig, ax = plt.subplots(figsize=(12, 10))

    # plot each digit class as a separate scatter
    for digit in range(10):
        mask = (labels == digit)
        ax.scatter(
            coords_2d[mask, 0],
            coords_2d[mask, 1],
            c=[COLOURS[digit]],
            label=f"Digit {digit}",
            alpha=0.4,  # transparency so overlapping points visible
            s=8,  # point size
            rasterized=True  # faster rendering for many points
        )

    # overlay centroids if provided
    if centroids_2d is not None:
        ax.scatter(
            centroids_2d[:, 0],
            centroids_2d[:, 1],
            c=COLOURS,
            s=200,  # large markers for centroids
            marker="*",  # star shape distinguishes from data
            edgecolors="black",
            linewidths=0.8,
            zorder=5,  # draw on top of data points
            label="_nolegend_"  # don't add to legend
        )
        # label each centroid with its digit
        for digit in range(10):
            ax.annotate(
                str(digit),
                (centroids_2d[digit, 0], centroids_2d[digit, 1]),
                fontsize=12,
                fontweight="bold",
                ha="center",
                va="bottom",
                xytext=(0, 8),
                textcoords="offset points"
            )

    # highlight specific points if provided (e.g. empty set examples)
    if highlight_indices is not None:
        ax.scatter(
            coords_2d[highlight_indices, 0],
            coords_2d[highlight_indices, 1],
            c="red",
            s=40,
            marker="x",
            linewidths=1.5,
            zorder=6,
            label="Flagged examples"
        )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Component 1", fontsize=11)
    ax.set_ylabel("Component 2", fontsize=11)
    ax.legend(loc="upper right", markerscale=2, fontsize=9,
              framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


# =============================================================================
# 7. GENERATE PLOTS
# =============================================================================

print("\nGenerating plots...")

# Plot 1: UMAP coloured by class
plot_embeddings(
    coords_2d=umap_2d,
    labels=labels_sample,
    title="UMAP Projection of MNIST CNN Embeddings (128D → 2D)\n"
          "Each point is one image. Colour = true digit class.",
    filename="umap_embeddings.png"
)

# Plot 2: UMAP with centroids overlaid
plot_embeddings(
    coords_2d=umap_2d,
    labels=labels_sample,
    title="UMAP Projection with Class Centroids (★)\n"
          "Stars show the mean embedding position for each digit class.",
    filename="umap_with_centroids.png",
    centroids_2d=centroids_2d_umap
)

# Plot 3: t-SNE coloured by class
plot_embeddings(
    coords_2d=tsne_2d,
    labels=labels_sample,
    title="t-SNE Projection of MNIST CNN Embeddings (128D → 2D)\n"
          "Note: distances between clusters are not meaningful in t-SNE.",
    filename="tsne_embeddings.png"
)

# =============================================================================
# 8. CENTROID DISTANCE HEATMAP
# =============================================================================
# Visualise the pairwise centroid distance matrix as a heatmap.
# Darker = closer in embedding space = more likely to appear in same
# prediction set. This is the visual version of the distance table
# printed by embedding_ci.py.

print("Generating centroid distance heatmap...")

centroid_distances = np.zeros((10, 10))
for i in range(10):
    for j in range(10):
        diff = centroids[i] - centroids[j]
        centroid_distances[i, j] = np.sqrt((diff ** 2).sum())

fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(centroid_distances, cmap="YlOrRd_r", aspect="auto")
# YlOrRd_r: yellow=far, red=close — red pairs are most likely to
# appear together in prediction sets

plt.colorbar(im, ax=ax, label="Euclidean Distance in Embedding Space")

# axis labels
ax.set_xticks(range(10))
ax.set_yticks(range(10))
ax.set_xticklabels([f"Digit {i}" for i in range(10)], rotation=45, ha="right")
ax.set_yticklabels([f"Digit {i}" for i in range(10)])

# annotate each cell with the distance value
for i in range(10):
    for j in range(10):
        ax.text(j, i, f"{centroid_distances[i, j]:.1f}",
                ha="center", va="center", fontsize=8,
                color="white" if centroid_distances[i, j] < centroid_distances.max() * 0.4
                else "black")

ax.set_title("Pairwise Centroid Distances in Embedding Space\n"
             "Red = close (likely to appear together in prediction sets)\n"
             "Yellow = far (unlikely to be confused)",
             fontsize=12, fontweight="bold", pad=15)

plt.tight_layout()
plt.savefig("centroid_distances.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: centroid_distances.png")

##================================================================
# 9. WHAT TO LOOK FOR IN PLOTS
###===============================================

print("\n" + "=" * 60)
print("WHAT TO LOOK FOR IN YOUR PLOTS")
print("=" * 60)
print("""
umap_embeddings.png:
  - Well separated clusters → CNN has learned distinct representations
  - Overlapping clusters between specific pairs (e.g. 4 and 9) →
    these are the digit pairs appearing together in prediction sets
  - Points sitting between clusters → likely empty set examples
  - Sub-clusters within one class → the class has multiple visual styles
    that a single centroid represents poorly

umap_with_centroids.png:
  - Is the centroid (★) in the middle of each cluster?
    If yes → centroid is a good representative
    If no  → the class has multiple sub-clusters, explaining why some
             examples are far from their centroid and get empty sets

tsne_embeddings.png:
  - Compare cluster shapes with UMAP — do the same pairs overlap?
  - t-SNE shows finer local structure within clusters
  - Ignore distances between clusters — not meaningful in t-SNE

centroid_distances.png:
  - Red pairs = most similar in embedding space = most likely to
    appear together in prediction sets
  - You should see 4↔9, 3↔8, 1↔7 as the closest pairs
  - For your financial documents this table will be the most
    interpretable output — showing which document types the model
    finds most semantically similar
""")

print("All plots saved. Open them in PyCharm by clicking the files")
print("in the project panel on the left.")
