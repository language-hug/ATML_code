import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# load data
input_file = "glove.twitter.27B.100d.names.pickle"

with open(input_file, 'rb') as f:
    embedding = pickle.load(f)

names = list(embedding.keys())
points = np.array([embedding[x] for x in names])
n, d = points.shape

# normalization
norms = np.linalg.norm(points, axis=1)
max_norm = np.max(norms)
print(f"Maximum norm before normalization: {max_norm}")

if max_norm > 1:
    points = points / max_norm
    norms = np.linalg.norm(points, axis=1)
    max_norm = np.max(norms)
    print(f"Maximum norm after normalization: {max_norm}")

# use PCA to reduce data to two dimensions
pca = PCA(n_components=2)
points_2d = pca.fit_transform(points)

# cost function
def compute_cost(points, centers):
    distances_squared = np.sum((points - centers[:, np.newaxis])**2, axis=2)
    return np.mean(np.min(distances_squared, axis=0))

# Non-private k-means algorithm from the teacher
def k_means(points, k, t):
    initial_assignment = np.random.choice(range(k), n)
    cluster_indexes = [ (initial_assignment == i) for i in range(k) ]
    cluster_sizes = [ cluster_indexes[i].sum() for i in range(k) ]

    for l in range(t):
        cluster_sums = [ np.sum(points[cluster_indexes[i]], axis=0) for i in range(k) ]
        centers = np.array([ cluster_sums[i] / max(1, cluster_sizes[i]) for i in range(k) ])
        distances_squared = np.sum((points - centers[:,np.newaxis])**2, axis=-1)
        assignment = np.argmin(distances_squared, axis=0)
        cluster_indexes = [ (assignment == i) for i in range(k) ]
        cluster_sizes = [ cluster_indexes[i].sum() for i in range(k) ]

    return centers, assignment

#
def compute_sigma(rho, t):
    sigma_squared = (3 * t) / rho
    sigma = np.sqrt(sigma_squared)
    return sigma
#private m_means algorithm
def m_means(points, k, t, rho):
    n, d = points.shape
    sigma = compute_sigma(rho, t)
    # initialize clusters
    initial_assignment = np.random.choice(range(k), n)
    cluster_indexes = [(initial_assignment == i) for i in range(k)]
    cluster_sizes = [cluster_indexes[i].sum() for i in range(k)]

    for l in range(t):
        # calculate the sum of clusters and add noise
        cluster_sums = [np.sum(points[cluster_indexes[i]], axis=0) for i in range(k)]
        centers = []
        n_counts = []
        for i in range(k):
            n_i = cluster_sizes[i]
            sum_x = cluster_sums[i]
            z = np.random.normal(0, sigma, size=d)
            c_i = (sum_x + z) / max(1, n_i)
            centers.append(c_i)
            # update cluster sizes and add noise
            z_prime = np.random.normal(0, sigma)
            n_i_noisy = n_i + z_prime
            n_counts.append(n_i_noisy)
        centers = np.array(centers)
        # assign data points to clusters
        distances_squared = np.sum((points - centers[:, np.newaxis])**2, axis=-1)
        assignment = np.argmin(distances_squared, axis=0)
        cluster_indexes = [(assignment == i) for i in range(k)]
        cluster_sizes = [cluster_indexes[i].sum() for i in range(k)]

    return centers, assignment


# Parameter settings

k = 5  # Number of clusters
t = 5  # Number of iterations

# Run non-private k-means and calculate cost
centers_non_private, assignment_non_private = k_means(points, k, t)
cost_non_private = compute_cost(points, centers_non_private)

# Define the range of the privacy parameter rho
rho_values = np.logspace(-3, 0, num=20)  # From 0.001 to 1 in log scale

# Run private m_means and calculate cost
costs = []
costs_std = []
num_runs = 5  # Number of runs for each rho value to calculate the average

for rho in rho_values:
    cost_runs = []
    for _ in range(num_runs):
        centers_private, assignment_private = m_means(points, k, t, rho)
        cost = compute_cost(points, centers_private)
        cost_runs.append(cost)
    average_cost = np.mean(cost_runs)
    std_cost = np.std(cost_runs)
    costs.append(average_cost)
    costs_std.append(std_cost)

# Plot the cost curve (Second figure)
plt.figure(figsize=(10, 6))
plt.errorbar(rho_values, costs, yerr=costs_std, fmt='-o', label='Private k-means')
plt.hlines(cost_non_private, xmin=rho_values[0], xmax=rho_values[-1], colors='r', linestyles='dashed', label='Non-private k-means')
plt.xlabel('Privacy Parameter ρ')
plt.ylabel('Cost')
plt.xscale('log')
plt.title('Clustering Cost vs Privacy Parameter ρ')
plt.legend()
plt.grid(True)
plt.show()

# Visualize clustering results for different rho values
selected_rhos = [0.001,0.01,0.1,0.5, 1]  # Select multiple rho values
num_plots = len(selected_rhos) + 1  # Including non-private clustering results
cols = 3  # Number of subplots per row
rows = 2  # Number of rows, adjust as needed
fig, axs = plt.subplots(rows, cols, figsize=(18, 10))
axs = axs.ravel()

# Plot non-private clustering results (First subplot)
centers_2d_non_private = pca.transform(centers_non_private)
axs[0].scatter(points_2d[:, 0], points_2d[:, 1], c=assignment_non_private, cmap='viridis', s=10)
axs[0].scatter(centers_2d_non_private[:, 0], centers_2d_non_private[:, 1], c='red', marker='x', s=100, label='Centers')
axs[0].set_title('Non-private k-means')
axs[0].legend()

# Plot private clustering results
for idx, rho in enumerate(selected_rhos):
    centers_private, assignment_private = m_means(points, k, t, rho)
    centers_2d_private = pca.transform(centers_private)
    axs[idx + 1].scatter(points_2d[:, 0], points_2d[:, 1], c=assignment_private, cmap='viridis', s=10)
    axs[idx + 1].scatter(centers_2d_private[:, 0], centers_2d_private[:, 1], c='red', marker='x', s=100, label='Centers')
    axs[idx + 1].set_title(f'Private k-means (ρ={rho})')
    axs[idx + 1].legend()

# Hide
for i in range(num_plots, rows * cols):
    fig.delaxes(axs[i])

plt.tight_layout()
plt.show()