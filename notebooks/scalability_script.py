"""
This contains the scripts needed to run the scalability experiments of the
paper. The results are saved in a directoy.
"""
import time
import os
import json

import numpy as np
import matplotlib.pyplot as plt
import torch as t

from diisco import DIISCO
import diisco.names as names

t.set_default_dtype(t.float64)

def sample_gaussian_process(length_scale : float, noise : float, start:int=0, end:int=10, n_time_points=100, timepoints=None) -> np.ndarray:
    if timepoints is not None:
        x = timepoints
        n_time_points = len(timepoints)
    else:
        x = np.linspace(start, end, n_time_points)

    cov = np.zeros((n_time_points, n_time_points))
    for i in range(n_time_points):
        for j in range(n_time_points):
            cov[i, j] = np.exp(-0.5 * (x[i] - x[j])**2 / length_scale**2)
            if i == j:
                cov[i, j] += noise

    mu = np.zeros(n_time_points)
    return x, np.random.multivariate_normal(mu, cov)


def generate_data(n_time_points: int, n_clusters: int, base_clusters: int, start:int=0, end:int=10, length_scale: float=1, noise: float=0.01) -> np.ndarray:
    dependent_clusters = n_clusters - base_clusters

    W_full = np.zeros((n_clusters, n_clusters, n_time_points))
    W_prior = np.zeros((n_clusters, n_clusters))
    # First we sample the time points
    time_points = np.random.uniform(start, end, n_time_points)
    time_points.sort()
    clusters = []
    for i in range(base_clusters):
        x, y = sample_gaussian_process(n_time_points=n_time_points, length_scale=length_scale, noise=0.0001, timepoints=time_points)
        clusters.append((x, y))
        W_full[i, i] = 1
        W_prior[i, i] = 1

    # Now we sample the dependent clusters
    for dep_num in range(dependent_clusters):
        # choose a random number of base clusters to use
        n_base_clusters_used = np.random.randint(1, base_clusters+1)
        # choose the base clusters to use
        base_clusters_used = np.random.choice(base_clusters, n_base_clusters_used, replace=False)

        x = time_points.copy()
        y = np.zeros(n_time_points)
        for base_clust in base_clusters_used:
            _, w = sample_gaussian_process(n_time_points=n_time_points, length_scale=length_scale * 10, noise=noise*0.0001, timepoints=time_points)
            W_full[dep_num + base_clusters, base_clust] = w
            W_prior[dep_num + base_clusters, base_clust] = 1
            y += clusters[base_clust][1] * w

        clusters.append((x, y))


    return time_points, clusters, W_full, W_prior


def run_model(n_time_points: int, save_dir: str, n_clusters: int, name):
    """
    Runs the model till convergence and saves the results
        :param n_time_points: The number of time points to use
        :param save_dir: The directory to save the results to

    """
    name = "{}_{}_{}_{}.json".format(name, n_time_points, n_clusters, "DiagonalNormal")
    print("Running", name)
    if os.path.exists(os.path.join(save_dir, name)):
        print("Skipping", name, "as it already exists")
        return
    base_clusters =  int((50/100) * n_clusters)
    start = 0
    end = 10 * (n_time_points / 100)

    timepoints, clusters, W_full, W_prior = generate_data(n_time_points, n_clusters, base_clusters, start=start, end=end)

    timepoints = t.tensor(timepoints, dtype=t.float64)
    timepoints = timepoints.view(-1, 1)
    proportions = [c[1] for c in clusters]
    proportions = np.array(proportions)
    proportions = t.tensor(proportions, dtype=t.float64).T
    W_prior = t.tensor(W_prior, dtype=t.float64)


    hyperparams = {
        names.LENGTHSCALE_F: 1,
        names.LENGTHSCALE_W: 10,
        names.SIGMA_F: 0.1,
        names.VARIANCE_F: 1,
        names.SIGMA_W: 0.1,
        names.VARIANCE_W: 1,
        names.SIGMA_Y: 0.1,
    }

    start_time = time.time()
    model = DIISCO(lambda_matrix=W_prior, hypers_init_vals=hyperparams, verbose=True)
    model.fit(timepoints,
            proportions,
            n_iter=50_000,
            patience=500,
            lr=0.007,
            hypers_to_optim=[],
            guide="DiagonalNormal")
    end_time = time.time()
    time_taken = end_time - start_time

    W_mean = model.sample(n_samples=10, n_samples_per_latent=100, timepoints=timepoints)[names.W].mean(dim=0).permute(1, 2, 0)

    diff = (W_full - W_mean.detach().numpy())**2
    diff = diff.mean(axis=-1)

    base = W_full**2
    base = base.mean(axis=-1) + 1e-10

    W_prior = W_prior.detach().numpy()
    r2 = (1 - diff/base) * W_prior

    # zero out nans
    r2[np.isnan(r2)] = 0
    r2 = r2.sum() / W_prior.sum()


    results = {
        "time_taken": time_taken,
        "hyperparams": hyperparams,
        "r2" : r2.item(),
        "mse" : diff.sum().item()/W_full.sum().item(),
    }


    # Save the results dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # use json to save the results
    with open(os.path.join(save_dir, name), "w") as f:
        json.dump(results, f)

    print("Done", name)


#################################################
# Main script
#################################################

if __name__ == "__main__":
    # Common directory to save all outputs
    save_dir = "results_multiple"

    # Define different configurations for each process to run
    n_time_points = [2, 5, 10, 20]
    n_clusters = [2, 5, 10, 20, 50, 100]

    configurations = []
    for run in range(10):
        for n in n_time_points:
            for c in n_clusters:
                configurations.append((n, f"{run}_save_dir_time", c, f"run_{n}_{c}"))
