import numpy as np
from scipy.stats import multivariate_normal
import random
import matplotlib.pyplot as plt

np.random.seed(seed=24)
random.seed(10)

# https://dhruvdakoria.hashnode.dev/mean-and-covariance-of-multivariate-data-with-python
# https://github.com/sumeyye-agac/expectation-maximization-for-gaussian-mixture-model-from-scratch/blob/main/EMforGMM.py


"""
this function creates 600 data items that origins from three bivariate normal distributions
"""
def create_data():
    m1 = [1, 1]  # consider a random mean and covariance value
    m2 = [7, 7]
    m3 = [20, 8]
    cov1 = [[3, 2], [2, 3]]
    cov2 = [[2, -1], [-1, 2]]
    cov3 = [[12, -11], [-11, 12]]
    x = np.random.multivariate_normal(
        m1, cov1, size=(200,)
    )  # Generating 200 samples for each mean and covariance
    y = np.random.multivariate_normal(m2, cov2, size=(200,))
    z = np.random.multivariate_normal(m3, cov3, size=(200,))
    d = np.concatenate((x, y, z), axis=0)
    return d, x, y, z


def logLikelihood(data, K, N, means, covariances, mixing_coefficients):
    likelihood = np.zeros((N, K))
    for n in range(N):
        for k in range(K):
            likelihood[n, k] = multivariate_normal.pdf(
                data[n], means[k], covariances[k]
            )
    log_likelihood = np.sum(np.log(likelihood.dot(mixing_coefficients)))
    return log_likelihood






def expectation_step(data, K, N, means, covariances, mixing_coefficients):
    denoms = []
    for k in range(K):
        denom = mixing_coefficients[k] * multivariate_normal.pdf(data, means[k], covariances[k])
        denoms.append(denom)
    denoms_= np.array(denoms)
    denoms_sum = np.sum(denoms_.T, axis=1)
    resps = denoms_ / denoms_sum
    return resps.T

def maximization_step(data, K, N, D, resps):
    data_ = data[:, np.newaxis]
    gamma_ = resps[:, :, np.newaxis]
    gamma_data_product = np.multiply(gamma_, data_)
    gamma_sum = np.sum(gamma_, axis=0)
    means_ = np.divide(np.sum(gamma_data_product, axis=0), gamma_sum)

    list_covs = []
    for k in range(K):
        var = data_[:, 0] - means_[k]
        d = gamma_[:, k] * var
        covs = var.T @ d
        list_covs.append(covs / np.sum(gamma_[:, k]))
    covs = np.array(list_covs)
    mix_coef = (gamma_sum / N).squeeze(1)
    return means_, covs, mix_coef


def prediction(data, K, N, means, covariances, mixing_coefficients):
    predicted_k = np.zeros(N, dtype=int)
    for n in range(N):
        probabilities = np.zeros(K)
        for k in range(K):
            probabilities[k] = mixing_coefficients[k] * multivariate_normal.pdf(
                data[n], means[k], covariances[k]
            )
        predicted_k[n] = int(np.argmax(probabilities))
    return predicted_k


def plot_state(data, means, covs, prediction_k):
    min = np.matrix(data).min(0).getA1()
    max = np.matrix(data).max(0).getA1()

    x1 = np.linspace(min[0], max[0], 200)
    x2 = np.linspace(min[1], max[1], 200)

    X, Y = np.meshgrid(x1, x2)

    Z1 = multivariate_normal(means[0], covs[0])
    Z2 = multivariate_normal(means[1], covs[1])
    Z3 = multivariate_normal(means[2], covs[2])

    # use colormap
    colormap = np.array(["r", "g", "b"])

    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    plt.figure(figsize=(10, 10))
    plt.scatter(data[:, 0], data[:, 1], marker="o", c=colormap[prediction_k])
    plt.contour(X, Y, Z1.pdf(pos), colors="r", alpha=0.5)
    plt.contour(X, Y, Z2.pdf(pos), colors="g", alpha=0.5)
    plt.contour(X, Y, Z3.pdf(pos), colors="b", alpha=0.5)
    plt.axis("equal")
    plt.xlabel("X-Axis", fontsize=16)  # X-Axis
    plt.ylabel("Y-Axis", fontsize=16)  # Y-Axis
    plt.legend()
    plt.grid()
    plt.show()


def em():
    # data = np.load("dataset.npy")  # path of data file
    all = create_data()
    data = all[0]
    N = data.shape[0]  # number of rows
    D = data.shape[1]  # number of columns -> dimension of data
    K = 3  # number of distributions/clusters
    max_iteration_number = 200  # number of maximum iteration
    convergence_value = 0.01  # negligible convergence value

    means = [
        [
            random.random(),
            random.random(),
        ],  # mean of dimension 1, dimension 2 for distribution 1
        [
            random.random(),
            random.random(),
        ],  # mean of dimension 1, dimension 2 for distribution 2
        [random.random(), random.random()],
    ]  # mean of dimension 1, dimension 2 for distribution 3

    covariances = [
        [[1.0, 0.0], [0.0, 1.0]],  # covariance matrix of distribution 1
        [[1.0, 0.0], [0.0, 1.0]],  # covariance matrix of distribution 2
        [[1.0, 0.0], [0.0, 1.0]],
    ]  # covariance matrix of distribution 3

    mixing_coefficients = np.zeros(K)
    mixing_coefficients[0] = 1 / K  # weight of distribution 1
    mixing_coefficients[1] = 1 / K  # weight of distribution 2
    mixing_coefficients[2] = 1 / K  # weight of distribution 3

    print("Initial Values: ")
    print("means = ", means)
    print("covariances = ", covariances)
    print("mixing_coefficients: ", mixing_coefficients)

    log_likelihoods = []
    current_log_likelihood = logLikelihood(
        data, K, N, means, covariances, mixing_coefficients
    )
    log_likelihoods.append(current_log_likelihood)

    i = 0
    while i < max_iteration_number:
        predicted_k = prediction(data, K, N, means, covariances, mixing_coefficients)
        plot_state(data, means, covariances, predicted_k)

        gamma = expectation_step(data, K, N, means, covariances, mixing_coefficients)
        means, covariances, mixing_coefficients = maximization_step(
            data, K, N, D, gamma
        )
        current_log_likelihood = logLikelihood(
            data, K, N, means, covariances, mixing_coefficients
        )
        log_likelihoods.append(current_log_likelihood)
        print("Iteration: ", i, " - Log likelihood value: ", current_log_likelihood)
        if abs(log_likelihoods[-1] - log_likelihoods[-2]) < convergence_value:
            break

        # Use if plots for some intermediate iterations are needed
        # if i%5==0:
        #    predicted_k = prediction(data, K, N, means, covariances, mixing_coefficients)
        #    plotting(i, data, predicted_k)

        i = i + 1

    predicted_k = prediction(data, K, N, means, covariances, mixing_coefficients)
    plot_state(data, means, covariances, predicted_k)
    #plotting(i, data, predicted_k)

    print("Final Values: ")
    print("means = ", means)
    print("covariances = ", covariances)
    print("mixing_coefficients: ", mixing_coefficients)

