# import libraries
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples, silhouette_score
import scipy.stats

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
%matplotlib inline

#TO DO - Create the parameters for your sample dataset

num_samples = 2000 #Create the sample dataset size  
num_k_clusters = 3   #Create clusters for the sample dataset
k_cluster_std = 1.5  #Create the standard deviation between clusters
random_state = 42  #Create the random state for reproducibility


#TO DO - Create the sample dataset using make_blobs() - see the link above

X, y = make_blobs(n_samples= num_samples, 
                  centers= num_k_clusters, 
                  cluster_std= k_cluster_std, 
                  random_state= random_state)

#Check the size of your sample set
print(X.shape, y.shape)

#TO DO
range_k_clusters = [2, 3, 4, 5, 6] #Create list of ints

kmeans_cluster_labels=[]  #This variable will hold the K-Means cluster predictions
kmeans_clusterer_list=[] #This variable will hold values needed to plot cluster centers
# Initialize the clusterer with k_clusters value from range_k_clusters

for k_clusters in range_k_clusters:
    
    #TO DO
    # Initialize KMeans cluster using n_clusters=k_clusters and random_state=rando_state
    kmeans_clusterer = KMeans(n_clusters=k_clusters, 
                              random_state=random_state)
    
    #TO DO
    # Fit dataset (X) to kmeans_clusterer
    kmeans_model = kmeans_clusterer.fit(X)
    
    #TO DO
    # Predict new dataset (X) clusters using the kmeans_model and predict()
    kmeans_prediction = kmeans_model.predict(X)
    
    #Add predictions to list called kmeans_cluster_labels
    kmeans_cluster_labels.append(kmeans_prediction)  
    
    #Add values to use to determine cluster centers for plotting
    kmeans_clusterer_list.append(kmeans_clusterer.cluster_centers_)
    
    gmm_cluster_labels=[]  #This variable will hold the GMM cluster predictions
gmm_clusterer_list =[] #This variable will hold values needed to plot cluster centers
gmm_clusterer_gen_list = [] #This variable will hold values needed to plot cluster centers


# Initialize the clusterer with k_clusters value from range_k_clusters

for k_clusters in range_k_clusters:

    #TO DO
    # Initialize Gaussian Mixture Model cluster using n_components=k_clusters and random_state=rando_state
    gmm_clusterer = GaussianMixture(n_components=k_clusters, 
                                    random_state=random_state)
    
    #TO DO
    # Fit dataset (X) to gmm_clusterer
    gmm_model = gmm_clusterer.fit(X)
    
    #TO DO
    # Predict new dataset (X) clusters using the gmm_model and predict()
    gmm_prediction =gmm_model.predict(X)
    
    #Add predictions to list called gmm_cluster_labels
    gmm_cluster_labels.append(gmm_prediction) 
    
    #Add values used to determine center of clusters when plotting
    gmm_clusterer_list.append(np.empty(shape=(gmm_clusterer.n_components, X.shape[1])))
    gmm_clusterer_gen_list.append(gmm_clusterer)
    
    
    # Silhouette Score gives the average value for all the samples.
for k_clusters, kmeans_cluster_label, gmm_cluster_label in\
    zip(range_k_clusters, kmeans_cluster_labels,gmm_cluster_labels):
    
    #TO DO
    #Create the Silhouette Score for X and the KMeans cluster (kmeans_cluster_label)
    silhouette_avg_kmeans = silhouette_score(X, kmeans_cluster_label)

    #Create the Silhouette Score for X and the GMM cluster (gmm_cluster_label)
    silhouette_avg_gmm = silhouette_score(X, gmm_cluster_label)
    
    print("For k_clusters =", k_clusters,
          "The avg silhouette_score using K-Means vs GMM is:", silhouette_avg_kmeans, "vs ", silhouette_avg_gmm,
          "\n")
    
    # Visualize and compare KMeans vs GMM Clusters for each value of k_clusters
#Variables centers1 and centers2 are used to plot the centers or each cluster
for k_clusters, kmeans_cluster_label, gmm_cluster_label, centers1, centers2, gmm_cluster in zip(range_k_clusters, kmeans_cluster_labels,gmm_cluster_labels,kmeans_clusterer_list, gmm_clusterer_list,gmm_clusterer_gen_list):

        
    # Plot showing the actual clusters formed
    # Create a subplot with 1 row and 2 features
    fig, [ax2,ax1] = plt.subplots(1,2)
    fig.set_size_inches(18, 7)
    
    #Plots for kmeans
    colors = cm.nipy_spectral(kmeans_cluster_label.astype(float) / k_clusters)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7, c=colors, edgecolor='k')

    # Labeling the clusters
    for i, c in enumerate(centers1):
        ax2.scatter(c[0], c[1], marker="o", alpha=1, s=50, color='r')

    ax2.set_title("The visualization of the K-Means clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    # plots for GMM
    colors = cm.nipy_spectral(gmm_cluster_label.astype(float) / k_clusters)
    ax1.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7, c=colors, edgecolor='k')
    
        
    # Measure the maximum density in GMM clusters
    for idx, i in enumerate(range(k_clusters)):
        density = scipy.stats.multivariate_normal(cov=gmm_cluster.covariances_[i], 
                                                  mean=gmm_cluster.means_[i]).logpdf(X)
        centers2[i, :] = X[np.argmax(density)]
    ax1.scatter(centers2[:, 0], centers2[:, 1], marker="o", alpha=1, s=50, color='r')
    

    ax1.set_title("The visualization of the GMM clustered data.")
    ax1.set_xlabel("Feature space for the 1st feature")
    ax1.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Scatter Plots of KMeans and GMM clustering on sample data "
                  "with k_clusters = %d" % k_clusters),
                 fontsize=14, fontweight='bold')

plt.show()

