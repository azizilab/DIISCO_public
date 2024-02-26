"""
This scipt contains the code to run the r2 vs prop experiment presented in the paper.
"""
import time
import os
import json
import numpy as np
import torch as t

from diisco import DIISCO
import diisco.names as names
import numpy as np
from multiprocessing import Pool

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

def run_and_compute_r2(W_full, W_prior, timepoints, clusters, hyperparams, name, save_dir, prop_data):
    if os.path.exists(os.path.join(save_dir, name)):
        print("Skipping", name, "as it already exists")
        return

    # We will use a subset of the data to speed up the computation
    n_used = int(prop_data * len(timepoints))
    points = np.random.choice(len(timepoints), n_used, replace=False)
    plain_name = name

    # sort the points
    points.sort()

    timepoints = t.tensor(timepoints, dtype=t.float64)
    timepoints = timepoints.view(-1, 1)
    proportions = [c[1] for c in clusters]
    proportions = np.array(proportions)
    proportions = t.tensor(proportions, dtype=t.float64).T
    W_prior = t.tensor(W_prior, dtype=t.float64)

    timepoints = timepoints[points]
    proportions = proportions[points]
    W_full = W_full[:, :, points]


    start_time = time.time()
    model = DIISCO(lambda_matrix=W_prior, hypers_init_vals=hyperparams, verbose=False)
    model.fit(timepoints,
            proportions,
            n_iter=50_000,
            patience=1_000,
            lr=0.0005,
            hypers_to_optim=[],
            guide="DiagonalNormal",
            subsample_size=int(0.7 * len(timepoints)))
    end_time = time.time()
    time_taken = end_time - start_time

    # This
    W_mean = model.sample(n_samples=101,n_samples_per_latent=10, timepoints=timepoints)[names.W].mean(dim=0).permute(1, 2, 0)


    unsqueezed_pior = W_prior.unsqueeze(-1)

    diff = (W_full - W_mean.detach().numpy())**2
    diff = diff.mean(axis=-1)

    base = W_full**2
    base = base.mean(axis=-1) + 1e-10

    W_prior = W_prior.detach().numpy()
    r2 = (1 - diff/base) * W_prior

    # zero out nans
    print(r2, np.isnan(r2))
    r2[t.isnan(r2)] = 0
    r2 = r2.sum() / W_prior.sum()


    results = {
        "time_taken": time_taken,
        "hyperparams": hyperparams,
        "r2" : r2.item(),
        "mse" : diff.sum().item()/W_full.sum().item(),
        "propo" : prop_data,
        "plain_name": plain_name,
        #"W_mean": W_mean.detach().numpy(),
    }

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # use json to save the results

    # pretty format prop_name
    prop_data = int(prop_data * 100)
    name = "prop_" + str(prop_data) + "_" + name
    with open(os.path.join(save_dir, name), "w") as f:
        json.dump(results, f)


    print("Done with", name)
    return results



def run_task(args):
    run, prop, W, W_prior, timepoints, clusters, hyperparams = args
    res = run_and_compute_r2(W, W_prior, timepoints, clusters, hyperparams, str(run), "prop_full", prop)
    return res['r2']



def main():

    # Define the time points
    n_time_points = 100
    timepoints = t.linspace(0, 100, n_time_points)
    W_prior = t.zeros(5, 5)
    W_prior[0, 0] = 1
    W_prior[1, 0] = 0
    W_prior[1, 1] = 1
    W_prior[2, 1] = 1
    W_prior[3, 0] = 1
    W_prior[3, 1] = 1
    W_prior[4, 2] = 1
    W_prior[4, 0] = 1


    # Define the weights according to the given equations
    W = t.zeros(5, 5, n_time_points)
    W[0, 0] = 1
    W[1, 1] = 1
    W[2, 1] = t.cos((timepoints - 10) / 30)
    W[3, 0] = t.cos((timepoints + 20) / 40)
    W[3, 1] = t.sin((timepoints + 20) / 40)
    W[4, 2] = t.cos((timepoints + 10) / 20)
    W[4, 0] = t.cos(timepoints / 30)
    W_full = W

    # Define the c values according to the given equations
    # Assuming ε1(t), ε2(t), etc. are defined elsewhere in your code
    c1 = 0.5 * t.cos((timepoints + 5) / 3) + t.cos(timepoints / 5) + t.randn((n_time_points,)) * 0.1
    c2 = t.sin((timepoints + 1) / 2) + t.cos(timepoints / 3) + t.cos((timepoints + 0.5) / 3) + t.randn((n_time_points,)) * 0.1
    c3 = W[2, 1] * c2 + t.randn((n_time_points,)) * 0.1
    c4 = W[3, 0] * c1 + W[3, 1] * c2 + t.randn((n_time_points,)) * 0.1
    c5 = W[4, 2] * c3 + W[4, 0] * c1 + t.randn((n_time_points,)) * 0.1


    clusters = [(timepoints, c1), (timepoints, c2), (timepoints, c3), (timepoints, c4), (timepoints, c5)]

    hyperparams = {
            names.LENGTHSCALE_F: 5,
            names.LENGTHSCALE_W: 20,
            names.SIGMA_F: 0.1,
            names.VARIANCE_F: 3,
            names.SIGMA_W: 0.01,
            names.VARIANCE_W: 2,
            names.LENGTHSCALE_W_RANGE: 10,
            names.SIGMA_Y: 0.1,
    }



    tasks = [(run, prop, W, W_prior, timepoints, clusters, hyperparams)
             for run in range(5)
             for prop in np.linspace(0.1, 1, 10)]

    #for task in tasks:
    #    run_task(task)

    # Create a pool of worker processes

    with Pool(processes=16) as pool:
        # Map run_task over the tasks, distributing them across the process pool
        results = pool.map(run_task, tasks)

        # Print the results
        for r2 in results:
            print(r2)

main()