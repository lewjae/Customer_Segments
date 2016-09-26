# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 16:15:56 2016

@author: Jae

Used unsupervised learning techniques to see if any similarities exist between customers, 
and how to best segment customers in distinct categories. One goal of this project is to 
best describe the variation in the different types of customers that a wholesale distributor 
interacts with. Doing so would equip the distributor with insight into how to best structure 
their delivery service to meet the needs of each customer.
 

"""
# Import libraries necessary for this project
import numpy as np
import pandas as pd
import renders as rs
from sklearn import cross_validation
from sklearn import tree
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import silhouette_score, make_scorer

# Load the wholesale customer dataset
try:
    data = pd.read_csv('customers.csv')
    data.drop(['Region','Channel'],axis = 1,inplace =True)
    print "Wholesale customers dataset has {} samples {} features each.".format(*data.shape)
except:
    print "Dataset could not be loaded. It the dataset missing?"

# Print a description of the dataset
print data.describe()

# Data Exploration
# Select indices of your choice you wish to sample from the dataset
indices = [61,100,220]

# Create a DataFrame of the chosen samples
samples = pd.DataFrame(data.loc[indices], columns = data.keys()).reset_index(drop =True)
print "Chosen samples of wholesale customers dataset:"
print samples
print
print "Chosen samples - data.mean"
print samples - data.mean()
print
print "Chosen samples - data.median"
print samples - data.median()

# Make a copy of the DataFrame, using the 'drop' function to drop the given feature
new_data = data.copy()
new_data = new_data.drop('Grocery',1)

# Split the data into training and testing sets using the given feature as the target
X_train, X_test, y_train, y_test = cross_validation.train_test_split(new_data, data.Grocery, test_size = 0.25,random_state =10)

# Create a decision tree regressor and fit it to the training set()
regressor = tree.DecisionTreeRegressor(min_samples_split = 10, random_state=10)
regressor.fit(X_train,y_train)

# Report the score od the prediction using the testing set
score = regressor.score(X_test,y_test)
print "R^2 score:", score

# Produce a scatter matrix for each pair of features in the data
pd.scatter_matrix(data,alpha =0.8, figsize = (20,12),diagonal = 'kde')

# Data Preprocessing
# Scale the data using the natueal algorithm - to make it as more like distribution
log_data = np.log(data.copy())

# Scale the sample data using the natural logarithm
log_samples = np.log(samples.copy())

# Product a scatter matrix for each pair of newly-transformed features
pd.scatter_matrix(log_data,alpha=0.8, figsize= (20,12), diagonal = 'kde')

# Display the log-transformed sample data
print log_samples
print
# Outlier Detection
outlierList =[]
# For each feature find the data points with extreme high or low values
for feature in log_data.keys():
    
    # Calculate Q1 (25th percentile of the data) for the given feature
    Q1 = np.percentile(log_data[feature],25)
    
    # Calculate q3 (75th percentile of the data) for the given feature
    Q3 = np.percentile(log_data[feature],75)
    
    # Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
    step = 1.5*(Q3-Q1)
    
    # Display the outliers
    
    print "Data points considered outlliers for the feature '{}':".format(feature)
    print (log_data[~((log_data[feature] >=Q1 -step) & (log_data[feature] <= Q3 +step))])
    outlier = log_data[~((log_data[feature] >=Q1 -step) & (log_data[feature] <= Q3 +step))].index.tolist()
    outlierList += outlier
     
print
print "Outliers list:", outlierList    

# Select the indices for data points you wish to remove
# Counter the element in the outliers list
outlierCounter = Counter(outlierList)
print "Outliers with counter:", outlierCounter
# Print out outliers without duplicates
print "Outliers list wo/duplicate:", sorted(list(outlierCounter))

# Pick the duplicates(index) in outlierList
outliersDuplicated = [i for i in outlierCounter if outlierCounter[i]>1]
print "Duplicated outliers ", sorted(outliersDuplicated)
outliers = outliersDuplicated

# Remove the outliers, if any were specified
good_data = log_data.drop(log_data.index[outliers]).reset_index(drop= True)

# Apply PCA to the good data with same number of dimensions as figures
pca = PCA()
pca.fit(good_data)
print "ex_plained_variance_ratio:", pca.explained_variance_ratio_

# Apply a PCA transformation to the sample log-data
pca_samples = pca.transform(log_samples)

# Generate PCA results plot
pca_results = rs.pca_results(good_data,pca)

# Print sample log-data after having a PCA transformation applied.
print pd.DataFrame(np.round(pca_samples,4), columns= pca_results.index.values)

# Fit PCA to the good data using only two dimensions
pca = PCA(n_components=2)

# Apply a PCA transformation the good data
pca.fit(good_data)
reduced_data = pca.transform(good_data)

# Apply a PCA transformation to the sample log-data
pca_samples = pca.transform(log_samples)

# Create a  DataFrame for the reduced data
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1','Dimension 2'])

# Display sample log-data after applying PCA transformation in two dimensions
print pd.DataFrame(np.round(pca_samples,4),columns = ['Dimension 1', 'Dimension 2'])

'''
# Apply your clustering algorithm of choice to the reduced data
kmeans = KMeans()
parameters = {'n_clusters':[2,3,4,5,6,7,8]}
my_scorer = make_scorer(silhouette_score,greater_is_better=True,pos_label = 'yes')
grid_kmean = GridSearchCV(kmeans,param_grid=parameters, scoring=my_scorer)
grid_kmean.fit(reduced_data)
preds = grid_kmean.predict(reduced_data)
score = silhouette_score(reduced_data,preds,random_state=10)
print "silhouette_score:", score
'''

# Apply your clustering algorithm,KMeans() to the reduced data
# Determine the optimal number of clustering based on Sihouette score
maxScore = -1
maxNc = 0
for nc  in range(2,10):
    clusterer = KMeans(n_clusters =nc)
    clusterer.fit(reduced_data)
    
    # Predict the cluster for each data point
    preds = clusterer.predict(reduced_data)
    score = silhouette_score(reduced_data,preds,random_state=10)
    print "Silhouette_score: {}, with n_comp = {}".format(score,nc)
    if score >= maxScore:
        maxNc = nc
        maxScore = score
print "Ideal number of cluster is:", maxNc

# Reapply KMeans() with optimal n_clusters
clusterer = KMeans(n_clusters = maxNc)
clusterer.fit(reduced_data)
preds = clusterer.predict(reduced_data)
    
# Preduct the cluster for each transformed sample data point
sample_preds = clusterer.predict(pca_samples)

# Find the clusters centers
centers = clusterer.cluster_centers_

# Display the results of the clusterng from implementation
print "centers: ",centers
rs.cluster_results(reduced_data,preds, centers, pca_samples)

# Inverse transform the centers
log_centers = pca.inverse_transform(centers)

# Take exponential function to move back normal scale
true_centers = np.exp(log_centers)

# Display the true centers
segments = ['Segment {}'.format(i) for i in range(0,len(centers))]
true_centers = pd.DataFrame(np.round(true_centers),columns = data.keys())
true_centers.index = segments
print "True_Centers:"
print true_centers
print "True_Centers - data.mean:"
print  true_centers - np.round(data.mean())
print "True_Centers - data.median"
print true_centers - np.round(data.median())

for i, pred in enumerate(sample_preds):
    print "Sample point", i, "predicted to be in Cluster", pred

print "true_centers - samples.ix[0]:"
print true_centers - samples.ix[0]
print "true_centers - samples.ix[1]:"
print true_centers - samples.ix[1]
print "true_centers - samples.ix[2]:"
print true_centers - samples.ix[2]

# Display the clustering results based on 'Channel' data
rs.channel_results(reduced_data,outliers,pca_samples)