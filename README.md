## The Expectation-maximization algorithm

This project entails the how and the why of the expectation-maximization algorithm used in the context of gaussian data. 

Clustering algorithms play an important role in understanding data. Grasp relationships and detect similarities among thousands of records in a dataset is a burdensome task. Fortunately, numerous algorithms come to aid in these matters. Generally, such algorithms make different assumptions about the underlying probabilistic generative process of the data.

The E-M algorithm maximizes the likelihood of data assuming a certain number distributions with different means and covariance matrices. GMM can be used for generative unsupervised learning or clustering. 

The E-M algorithm is able to fit data to distributions as in the images below.

<img src="https://github.com/user-attachments/assets/3010d96f-9ee0-4398-9ebb-26a3d68ed8de" width="400" >

<img src="https://github.com/user-attachments/assets/9e977c85-7298-4b2c-9d18-c6e4433eba92" width="400" >

### Similarities with  K-means clustering
The EM algorithm has similarities with K-means clustering. EM not only uses the means, but also the spread of data to cluster data.
* Randomly initialize K cluster centroids μₖ.
* Compute the distance of each data point with all K cluster centroids.
* Assign the data point to the closest cluster centroid (one with minimum distance).
* Re-compute the centroid for each K cluster, based on the points inside the cluster.
* Repeat the above steps for a fixed number of iterations or until convergence is reached.

### Two steps
Initially, distributions can be created using random values for means and covariance matrix. But after that, the expectation step and the maximization step will be applied iteratively.
#### The expectation step
In the expectation step, the algorithm computes the probability that each data point belongs to each distribution of the GMM.
For a simple normal distribution chances for a result can be obtained using the pdf function of a distribution. When setting 2 clusters, probabilities for every data point (x) are obtained for every distribution. This could be done like this:

  `from scipy.stats import norm`
  
  `p1=norm.pdf(x1, mean, sigma)`
  
  `p2=norm.pdf(x1, mean, sigma)`

For every datapoint chances for both distributions are summed.
When using 2 normal distributions norm(0,1) and norm(1,1) as centroids for the clusters, for one point 2 chances can be obtained:

    `x1=2`
    `p1 = norm.pdf(x1,0,1)`
    `p2 = norm.pdf(x1,1,1)`

Responsibilities can be calculated
    $respcluster1 = \frac{p1}{p1+p2}$
  
    

#### The maximization step




