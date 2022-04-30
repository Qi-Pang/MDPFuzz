import numpy as np
import copy
from scipy.stats import multivariate_normal
from tqdm import tqdm
import scipy

def generate_MoG_data(num_data, means, covariances, weights):
    """ Creates a list of data points """
    num_clusters = len(weights)
    data = []
    for i in tqdm(range(num_data)):
        #  Use np.random.choice and weights to pick a cluster id greater than or equal to 0 and less than num_clusters.
        k = np.random.choice(len(weights), 1, p=weights)[0]
        # Use np.random.multivariate_normal to create data from this cluster
        x = np.random.multivariate_normal(means[k], covariances[k])
        data.append(x)
    return data

def log_sum_exp(Z):
    """ Compute log(\sum_i exp(Z_i)) for some array Z."""
    return np.max(Z) + np.log(np.sum(np.exp(Z - np.max(Z))))

def loglikelihood(data, weights, means, covs):
    """ Compute the loglikelihood of the data for a Gaussian mixture model with the given parameters. """
    num_clusters = len(means)
    num_dim = len(data[0])
    
    ll = 0
    for d in data:
        
        Z = np.zeros(num_clusters)
        for k in range(num_clusters):
            
            # Compute (x-mu)^T * Sigma^{-1} * (x-mu)
            delta = np.array(d) - means[k]
            exponent_term = np.dot(delta.T, np.dot(np.linalg.inv(covs[k]), delta))
            
            # Compute loglikelihood contribution for this data point and this cluster
            Z[k] += np.log(weights[k])
            Z[k] -= 1/2. * (num_dim * np.log(2*np.pi) + np.log(np.linalg.det(covs[k])) + exponent_term)
            
        # Increment loglikelihood contribution of this data point across all clusters
        ll += log_sum_exp(Z)
        
    return ll

def EM(data, init_means, init_covariances, init_weights, maxiter=10, thresh=1e-3):
    # Make copies of initial parameters, which we will update during each iteration
    means = init_means[:]
    covariances = init_covariances[:]
    weights = init_weights[:]
    # Infer dimensions of dataset and the number of clusters
    num_data = len(data)
    num_dim = len(data[0])
    num_clusters = len(means)
    
    # Initialize some useful variables
    resp = np.zeros((num_data, num_clusters))
    ll = loglikelihood(data, weights, means, covariances)
    ll_trace = [ll]
    
    for i in range(maxiter):
        if i % 5 == 0:
            print("Iteration %s" % i)
        
        # E-step: compute responsibilities
        # Update resp matrix so that resp[j, k] is the responsibility of cluster k for data point j.
        # Hint: To compute likelihood of seeing data point j given cluster k, use multivariate_normal.pdf.
        for j in range(num_data):
            for k in range(num_clusters):
                # YOUR CODE HERE
                resp[j, k] = weights[k]*multivariate_normal.pdf(data[j],means[k],covariances[k])
        row_sums = resp.sum(axis=1)[:, np.newaxis]
        resp = resp / row_sums # normalize over all possible cluster assignments

        # M-step
        # Compute the total responsibility assigned to each cluster, which will be useful when 
        # implementing M-steps below. In the lectures this is called N^{soft}
        counts = np.sum(resp, axis=0)
        
        for k in range(num_clusters):
            
            # Update the weight for cluster k using the M-step update rule for the cluster weight, \hat{\pi}_k.
            # YOUR CODE HERE
            weights[k] = counts[k]/num_data
            # Update means for cluster k using the M-step update rule for the mean variables.
            # This will assign the variable means[k] to be our estimate for \hat{\mu}_k.
            weighted_sum = 0
            for j in range(num_data):
                # YOUR CODE HERE
                weighted_sum += (resp[j,k]*data[j])
            # YOUR CODE HERE
            means[k] = weighted_sum/counts[k]
            
            # Update covariances for cluster k using the M-step update rule for covariance variables.
            # This will assign the variable covariances[k] to be the estimate for \hat{\Sigma}_k.
            weighted_sum = np.zeros((num_dim, num_dim))
            for j in range(num_data):
                # YOUR CODE HERE (Hint: Use np.outer on the data[j] and this cluster's mean)
                weighted_sum += (resp[j,k]*np.outer(data[j]-means[k],data[j]-means[k]))
            # YOUR CODE HERE
            covariances[k] = weighted_sum/counts[k]
          
        
        # Compute the loglikelihood at this iteration
        # YOUR CODE HERE
        ll_latest = loglikelihood(data, weights, means, covariances)
        ll_trace.append(ll_latest)
        
        # Check for convergence in log-likelihood and store
        if (ll_latest - ll) < thresh and ll_latest > -np.inf:
            break
        ll = ll_latest
    
    if i % 5 != 0:
        print("Iteration %s" % i)
    
    out = {'weights': weights, 'means': means, 'covs': covariances, 'loglik': ll_trace, 'resp': resp}

    return out


def online_em(K, dataX, S, weights, means, covariances, iteration_num):
    gamma = 1 / (iteration_num + 10)
    w = np.zeros(K)
    for k in range(K):
        w[k] = weights[k] * multivariate_normal.pdf(dataX, means[k], covariances[k])
    w += 1e-5
    w = w / np.sum(w)
    new_S = copy.deepcopy(S)
    for i in range(K):
        new_S[i][0] = S[i][0] + gamma * (w[i] - S[i][0])
        
        new_S[i][1] = S[i][1] + gamma * (w[i]*dataX - S[i][1])
        new_S[i][2] = S[i][2] + gamma * (w[i]*np.matmul(dataX.T, dataX) - S[i][2])

    for i in range(K):
        weights[i] = new_S[i][0]
        means[i] = new_S[i][1] / new_S[i][0]
        covariances[i] = (new_S[i][2] - np.matmul(means[i:i+1].T, new_S[i][1])) / new_S[i][0]
        W, V = np.linalg.eigh(covariances[i])
        W = np.maximum(W, 1e-5)
        D = np.diag(W)
        reconstruction = np.matmul(np.matmul(V, D), np.linalg.inv(V))
        covariances[i] = reconstruction

    return weights, means, covariances, new_S


if __name__ == '__main__':
    K = 10
    dimension = 17
    init_means = np.ones((K, dimension))
    for i in range(1,K):
        init_means[i, :] = 1/i
    # init_covariances = np.random.random(size=(K, dimension, dimension))
    init_covariances = [np.eye(dimension)] * K
    init_weights = np.ones(K) / K
    np.random.seed(4)
    data = generate_MoG_data(1000, init_means, init_covariances, init_weights)
    data = np.array(data)
    init_covariances = [np.cov(data.T)] * K
    print('start EM')
    S = []
    for i in range(K):
        temp = dict()
        temp[0] = 1 / K
        temp[1] = temp[0] * np.mean(data[i:i+15], axis=0)
        temp[2] = np.zeros((data.shape[1], data.shape[1]))
        for j in range(i, i+15):
            temp[2] += temp[0] * np.matmul(data[j: j+1].T, data[j: j+1])
        temp[2] /= 15
        S.append(temp)

    weights = np.zeros(K)
    means = np.zeros((K, data.shape[1]))
    covariances = np.zeros((K, data.shape[1], data.shape[1]))
    for i in range(K):
        weights[i] = S[i][0]
        means[i] = S[i][1] / S[i][0]
        covariances[i] = np.eye(data.shape[1])

    for i in tqdm(range(1000)):
        if i % 2 == 0:
            weights, means, covariances, S = online_em(K, data[15:16], copy.deepcopy(S), copy.deepcopy(weights), copy.deepcopy(means), copy.deepcopy(covariances), i+1)
        else:
            weights, means, covariances, S = online_em(K, data[16:17], copy.deepcopy(S), copy.deepcopy(weights), copy.deepcopy(means), copy.deepcopy(covariances), i+1)