#Here's the code to predict labels:

from sklearn import datasets, cluster
# load data
X = datasets.load_iris().data[:10]

# Spefify the parameters for the clustering. 
# 'ward' linkage is default. Can also use 'complete' or 'average'.
clust = cluster.AgglomerativeClustering(n_cluster=3, linkage='ward')
labels = clust.fit_predict(X)

# 'labels' now contains an array representing which cluster 
# each point belongs to:
# [1 0 0 0 1 2 0 1 0 0]

#Here's the code to draw the dendrogram
from scipy.cluster.hierarchy import dendrogram, ward, single
from sklearn import datasets
import matplotlib.pyplot as plt

# Load dataset
X = datasets.load_iris().data[:10]

# Perform clustering
linkage_matrix = ward(X)

# plot dendrogram
dendrogram(linkage_matrix)
plt.show()
