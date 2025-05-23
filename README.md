## The Expectation-maximization algorithm

This project entails the how and the why of the expectation-maximization algorithm used in the context of gaussian data. 

Clustering algorithms play an important role in understanding data. To grasp relationships and detect similarities among thousands of records in a dataset is a burdensome task. Fortunately, numerous algorithms come to aid in these matters. Generally, such algorithms make different assumptions about the underlying probabilistic generative process of the data.

The E-M algorithm maximizes the likelihood of data, assuming a certain number distributions with different means and covariance matrices. GMM can be used for generative unsupervised learning or clustering. 

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
Initially, distributions can be created using random values for means and variance matrix. But after that, the expectation step and the maximization step will be applied iteratively. For the sake of simplicity the distributions will be univariate normal distributions initially having means 0 and 1 and sigmas 1 and 1. These distributions and the samples [0,1,3,4,5,6,2,4.5,7,8,10,12] will be subject of the example. Data is colored in the distribution it belongs most to.

![distri_0](https://github.com/user-attachments/assets/3cd2f931-4070-4606-9b51-35cdaef824e0)


#### The expectation step
In the expectation step, the algorithm computes the probability that each data point belongs to each distribution of the GMM.
For a simple normal distribution chances for a result can be obtained using the pdf function of a distribution. When setting 2 clusters, probabilities for every data point (x) are obtained for every distribution. Initially when using 2 clusters, every clusters is supposed to weigh 0.5. Generating probabilities for values from the sample could be done like this:

  `from scipy.stats import norm`

  ` possibilityCluster1=05`

  ` possibilityCluster2=05`
  
  `likelihood1= possibilityCluster1*norm.pdf(x1, mean, sigma)`
  
  `likelihood2= possibilityCluster2*norm.pdf(x1, mean, sigma)`

For every datapoint chances for both distributions are summed.
When using two normal distributions C1=norm(0,1) and C2=norm(1,1) as 'centroids' for the clusters, for one point 2 chances can be obtained:

    `x1=2`
    `likelihood1 = possibilityCluster1*norm.pdf(x1,0,1)`
    `likelihood2 = possibilityCluster2*norm.pdf(x1,1,1)`

Responsibilities can be calculated as follows

$likelihoodCluster1 = \frac{likelihood1}{likelihood1+likelihood2}$

$likelihoodCluster2 = \frac{likelihood2}{likelihood1+likelihood2}$

When plugging in the numbers likelihoodCluster1 = 0.18 and likelihoodCluster2=0.82. These chances (which add up to 1) express the likelihood a number is part of one of both clusters. For every datapoint in the dataset, these cluster likelihoods are calculated. For some dataset these numbers calculated and rounded.

![likeli_it_0](https://github.com/user-attachments/assets/8c43dea1-ddf1-43c7-914d-a0cbaf3fd438)

#### The maximization step

In the maximization step the scaled likelihoods are used to calculate means and sigmas for the clusters. A new mu (expected value) for Cluster1 and Cluster2 is being calculated using the scaledLikelihoods1(13 numbers) and the samples(13 nummbers). The more likely a number from the sample is in a cluster, the more its value weighs in the new mu. This way the new mu will shift towards its 'real' value. 

`muCluster1 = np.sum(samples * scaledLikelihoods1)`

`muCluster2 = np.sum(samples * scaledLikelihoods2)`

The updated mu values will be used to calculate new variances. The scaledLikelihoods will guarantee that more likely samples will weigh more in the newly calculated variances.

`varianceCluster1 = np.sum(((samples - muCluster1) ** 2) * scaledLikelihoods1)/numberOfSamples`

`varianceCluster2 = np.sum(((samples - muCluster2) ** 2) * scaledLikelihoods2)/numberOfSamples`

For Cluster 1 values look like this, the new mean is 0.925 and the new sigma value is 0.37, the square root of the variance.

![mu_sigma_it_0](https://github.com/user-attachments/assets/137557ad-64dd-49c9-a0cc-08dd849fcaf9)

For Cluster 2 values would be calculated likewise.

At the end of this first step, distributions and data looks like this.

![distri_1](https://github.com/user-attachments/assets/bd625f77-19a5-49ae-bb09-4f663d09ebb5)

The only thing that needs to be recalculated is the chance to belong to the different clusters (initially they chances were set to 0.5 and 0.5), the normalized likelihoods for Cluster1 are summed up and divided by the number of samples. So the chance to belong to Cluster 1 is 0.10 and the chance to belong to Cluster 2 is 0.90.

#### End of E-M algorithm

The algorithm stops when the overall likelihood gain of an iteration  is below some threshold.
The overall likelihood for the samples given two distributions is the product of every sum of two independent clusterlikelihoods of every item in the samples.

  `likelihood1 = possibilityCluster1*norm.pdf(samples,mu1,sigma1)`
  
  `likelihood2 = possibilityCluster2*norm.pdf(samples,mu2,sigma2)`
  
  `overallLogLikelihood = np.log(prod(likelihood1 + likelihood2))`

![distri_9](https://github.com/user-attachments/assets/67927e67-4a5d-48df-a07e-e25041335551)

As this example is very simple, the model is converged after a couple of iterations.

![likelihoods](https://github.com/user-attachments/assets/9b506b53-328e-4211-910a-fbd10c12acb2)


### The multivariate case

In the bivariate case the normal distributions have two means and a 2 by 2 covariance matrix. This complicates matters.

