## The Expectation-maximization algorithm

This project entails the how and the why of the expectation-maximization algorithm used in the context of gaussian data. 

Clustering algorithms play an important role in understanding data. Grasp relationships and detect similarities among thousands of records in a dataset is a burdensome task. Fortunately, numerous algorithms come to aid in these matters. Generally, such algorithms make different assumptions about the underlying probabilistic generative process of the data.

The E-M algorithm maximizes the likelihood of data assuming a certain number distributions with different means and covariance matrices. GMM can be used for generative unsupervised learning or clustering. 

The E-M algorithm is able to fit data to distributions as in the images below.

<img src="https://github.com/user-attachments/assets/3010d96f-9ee0-4398-9ebb-26a3d68ed8de" width="400" >

<img src="https://github.com/user-attachments/assets/9e977c85-7298-4b2c-9d18-c6e4433eba92" width="400" >

To explain how the algorithm works, I will use simple data that will end up in 2 univariate clusters.

### Similarities with  K-means clustering

The EM algorithm has similarities with K-means clustering. EM not only uses the means, but also the spread of data to cluster data.
* Randomly initialize K cluster centroids μₖ.
* Compute the distance of each data point with all K cluster centroids.
* Assign the data point to the closest cluster centroid (one with minimum distance).
* Re-compute the centroid for each K cluster, based on the points inside the cluster.
* Repeat the above steps for a fixed number of iterations or until convergence is reached.

### Two steps
Initially, distributions can be created using random values for means and variance matrix. But after that, the expectation step and the maximization step will be applied iteratively.
#### The expectation step
In the expectation step, the algorithm computes the probability that each data point belongs to each distribution of the GMM.
For a simple normal distribution chances for a result can be obtained using the pdf function of a distribution. When setting 2 clusters, probabilities for every data point (x) are obtained for every distribution. Initially when using 2 clusters, every clusters is supposed to weigh 0.5. Generating probabilities for values from the sample could be done like this:

  `from scipy.stats import norm`

  `w1=05`

  `w2=05`
  
  `p1=w1*norm.pdf(x1, mean, sigma)`
  
  `p2=w2*norm.pdf(x1, mean, sigma)`

For every datapoint chances for both distributions are summed.
When using two normal distributions C1=norm(0,1) and C2=norm(1,1) as 'centroids' for the clusters, for one point 2 chances can be obtained:

    `x1=2`
    `p1 = w1*norm.pdf(x1,0,1)`
    `p2 = w2*norm.pdf(x1,1,1)`

Responsibilities can be calculated as follows

$likelihoodCluster1 = \frac{p1}{p1+p2}$

$likelihoodCluster2 = \frac{p2}{p1+p2}$

When plugging in the numbers likelihoodCluster1 = 0.18 and likelihoodCluster2=0.82. These chances (which add up to 1) express the likelihood a number is part of one of both clusters. For every datapoint in the dataset, these cluster likelihoods are calculated. For some dataset these numbers calculated and rounded.

![likeli_it_0](https://github.com/user-attachments/assets/8c43dea1-ddf1-43c7-914d-a0cbaf3fd438)

#### The maximization step

In the maximization step the scaled likelihoods are used to calculate means and sigmas for the clusters. A new mu (expected value) for Cluster1 and Cluster2 is being calculated using the scaledLikelihoods1(13 numbers) and the samples(13 nummbers). The more likely a number from the sample is in a cluster, the more its value weighs in the new mu. This way the new mu will shift towards its 'real' value. 

`muCluster1 = sum(samples * scaledLikelihoods1)`

`muCluster2 = sum(samples * scaledLikelihoods2)`

The updated mu values will be used to calculate new variances. The scaledLikelihoods will guarantee that more likely samples will weigh more in the newly calculated variances.

`varianceCluster1 = sum(((samples - muCluster1) ** 2) * scaledLikelihoods1)`

`varianceCluster2 = sum(((samples - muCluster2) ** 2) * scaledLikelihoods2)`

For Cluster 1 values look like this, the new mean is 0.925 and the new sigma value is 1.34

![mu_sigma_it_0](https://github.com/user-attachments/assets/137557ad-64dd-49c9-a0cc-08dd849fcaf9)

For Cluster 2 values would be calculated likewise.























