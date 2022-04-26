import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from helper_functions import fit_random_forest_classifier, plot_rp, show_images_by_digit

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

train = pd.read_csv('./data/train.csv')
train.fillna(0, inplace=True)

# save the labels to a Pandas series target
y = train['label']

# Drop the label feature
X = train.drop("label",axis=1)

show_images_by_digit(2) # Try looking at a few other digits

# TODO: performs random projection to reduce the dataset dimension. 
# To start, use 0.5 as the epsilon value.

from sklearn.random_projection import SparseRandomProjection
rp = SparseRandomProjection(eps = 0.5) 
X_rp = rp.fit_transform(X)

rp_dim = rp.n_components_
X_dim = X.shape[1]
print("The orignial data has {} dimensions and it is reduced to {} after random projection.".format(X_dim, rp_dim))

# Todo: fit the new data with a classifier
fit_random_forest_classifier(X_rp, y)

# TODO: write a loop to transform X using different epsilon
for sample_eps in np.arange(0.5, 1,0.2):
    rp = SparseRandomProjection(eps = sample_eps) 
    X_rp = rp.fit_transform(X)
    
    acc = fit_random_forest_classifier(X_rp, y)
    print("With epsilon = {:.2f}, the transformed data has {} components, a random forest acheived an accuracy of {}.".format(sample_eps, X_rp.shape[1], acc))
    
    # Calulate the number of components with varying eps
from sklearn.random_projection import johnson_lindenstrauss_min_dim
eps = np.arange(0.1, 1, 0.01)
n_comp = johnson_lindenstrauss_min_dim(n_samples=1e6, eps=eps)

plt.plot(eps, n_comp, 'bo');
plt.xlabel('eps');
plt.ylabel('Number of Components');
plt.title('Number of Components by eps');

# TODO: find the number of samples and components in X
X_sample, X_comp = X.shape
print("The orignial data has {} samples with dimension {}.".format(X_sample, X_comp))

# TODO: define a n_components and perform random projection on data X.
# Store the transformed data in a `X_rp` variable
n_components = 30

rp = SparseRandomProjection(n_components=n_components)
X_rp = rp.fit_transform(X)

plot_rp(X, X_rp, n_components)

