
"""
Importing all the relevant modules
"""

# Pandas to load CSV file and perform some basic data pre-processing operations
# e.g dropping columns, checking for null values
import pandas as pd

# Numpy for vector and matrix operations
import numpy as np

# DecisionTreeRegressor to train and predict the missing values from the data
# Which we will be intentionally creating in our IRIS dataset
from sklearn.tree import DecisionTreeRegressor

# KMeans will help us cluster/group the similar/alike data 
from sklearn.cluster import KMeans

# For Normalization
from sklearn.preprocessing import MinMaxScaler

# mean_squared_error utility will help us quantify our DecisionTree training results 
# and how close to accurate they are
from sklearn.metrics import mean_squared_error

# Input encoder, to transform categorical values into numerical data
from sklearn.preprocessing import LabelEncoder

# To generate a frequency distribution of data
# e.g total count of elements in different clusters
import collections






# ### Loading data into pandas dataframe 

print("Loading IRIS dataset into pandas dataframe\n")
df_iris = pd.read_csv("./Data/iris.data", header=None)
print(f"Shape {df_iris.shape}")
print(df_iris.head(11))


# Display some basics statistical information from the data
print("\nStatistical Information regarding data\n")
print(df_iris.describe())


# ### Checking if there are any null values in the given dataset, because they might affect the performance of both of our DecisionTreeRegressor and KMean Algorithms

if(df_iris.isna().sum().sum() == 0):
    print("\nNo missing values\n")
else:
    print("\nMissing values in data detected\n")


# ### Transforming categorical labels into numerical data, this will be helpful for accuracy checks
labels = df_iris.drop([0, 1, 2, 3], axis=1)

encoder = LabelEncoder()
encoder.fit(labels[4].values) # Using 4 as Column, Index because even dropping, pandas maintain's the remaining data indices
encoded_labels = encoder.transform(labels[4].values)
print("\nEncoding Categorical data into numeric data\n")
print(encoded_labels)

print("\n Label Classes, assigned by LabelEncoder\n")
print(encoder.classes_)

counts = collections.Counter(encoded_labels)
print("\nOriginal Cluster Member Counts\n")
print(f"Cluster-0 member count = {counts[0]}")
print(f"Cluster-1 member count = {counts[1]}")
print(f"Cluster-2 member count = {counts[2]}")


# ### Drop label column from primary data, since its not directly useful in KMeans
# ### Although can be used for accuracy check later on

# Dropping categorical label name, because in KMean it does not serve us any purpose
df_iris.drop([4],axis=1, inplace=True)

print("\nPrinting IRIS data after `Label/categorical label` column drop\n")
print(df_iris.head(11))


# #  Assignment Section 1, DecisionTreeRegressor

series_original_Y_testing_data = [] # Will contain the original non-smudged values of testing data
                                             # To be later used for calculating regressor mean_squared_error

# ###  Missing Every 5th row in the dataset, as required by the assignment

for (i, row) in df_iris.iterrows():
    if((i % 5 == 0) and i != 0): # Miss every 5th row except first one
        series_original_Y_testing_data.append(df_iris.at[i, 0])
        df_iris.at[i, 0] = np.NaN
        
print(f"\n (Missing Values) Rows with NaN  in first Column  = {df_iris[0].isna().sum()}")
print(f" (Regular Values) Rows with data in first Column = {len(df_iris)- df_iris[0].isna().sum()}")
print("\n Printing dataframe with missing data in column # 0")
print(df_iris.head(21))


# ### Splitting data into training and testing for regressor algorithm


df_training = df_iris[np.isnan(df_iris[0]) == False]
df_testing  = df_iris[np.isnan(df_iris[0])]

print(f"\n Original IRIS dataset value count = {len(df_iris)}")
print(f" Value count for training data = {df_training.shape}")
print(f" Value count for testing data = {df_testing.shape}")



print("\nPrinting training dataset\n")
print(df_training.head())

print("\nPrinting testing dataset\n")
print(df_testing.head())


# ### Now further segregating data into Predictors and Ground truth for both training and testing data


# Data to be used for training DecisionTreeRegressor
X_training_predictors   = df_training[[1,2,3]]
Y_training_ground_truth = df_training[0]

# Data to be predicted by the regressor
x_testing_predictors    = df_testing[[1,2,3]]
y_testing_ground_truth  = df_testing[0]


# ## Training DecisionTreeRegressor


# Training a DecisionTreeRegressor with training data
regressor = DecisionTreeRegressor()
regressor = regressor.fit(X_training_predictors, Y_training_ground_truth)

# ###  Predicting missing values of testing data using the trained regressor

for (i, row) in x_testing_predictors.iterrows():
    y_testing_ground_truth.at[i] = regressor.predict([row])
print("\nPredicted Missing values")
print(y_testing_ground_truth.head())


# ### Restuffing predicted data back into the missing columns of original data

# Putting data back into original dataframe

# This index is being fetched from the y_testing_ground_truth
# Because pandas preserved the origianl index locations of the smudged data into the 
# Main IRIS dataframe, hence we can simply re-access those indices and put the predicted data

for special_index in y_testing_ground_truth.index:
    df_iris.at[special_index, 0] =  y_testing_ground_truth[special_index]

# Now printing the orignal IRIS dataset with the missing values fixed
print("\nIRIS dataframe after merging predicted data back with training data\n")
print(df_iris.head(21))


# ### Calculating Accuracy of our DecisionTreeRegressor predictions

# Calculating Mean Squared Error between the original, non-smudged data and the predicted data
err = mean_squared_error(series_original_Y_testing_data, y_testing_ground_truth)
print(f"\nRegression Mean-Squared-Error {err}\n")


# # Assignment Section 2, Clustering 

# Performing Normalization


df_iris_non_normalized = df_iris.copy(deep=True) # Perform a deep copy of IRIS, we will used this cloned copy
                                                 # For running KMeans on non-normalized data
    
print("\nStandard Final IRIS dataframe without any normalization\n")
print(df_iris_non_normalized.head(6))
    
# Apply normalization on the other dataframe    
    
scalar = MinMaxScaler()
scalar.fit(df_iris)

df_iris = scalar.transform(df_iris)
#Recreating dataframe because MinMaxScalar.transform returns numpy data structure
df_iris = pd.DataFrame(df_iris) 
print("\nNormalized IRIS dataframe\n")
print(df_iris.head(6))


K = 3 # Keeping K 3 since there are only 3 species IRIS in dataset, 
      # hence for this toy dataset it is a relatively straight-forword decision
iterations = 10000    

k_cluster = KMeans(n_clusters=K, max_iter=iterations, random_state=11)
k_cluster = k_cluster.fit(df_iris)

print("\nClusters generated by KMeans (With Normalization)\n")
print(k_cluster.labels_)


counts = collections.Counter(k_cluster.labels_)

print("Cluster Member Counts (With Normalization)\n")
print(f"Cluster-0 member count = {counts[0]}")
print(f"Cluster-1 member count = {counts[1]}")
print(f"Cluster-2 member count = {counts[2]}")

# Now Clustering with non-normalized data

k_cluster_not_normalized = KMeans(n_clusters=K, max_iter=iterations, random_state=11)
k_cluster_not_normalized = k_cluster_not_normalized.fit(df_iris_non_normalized)


print("\nClusters generated by KMeans\n")
print(k_cluster_not_normalized.labels_)


counts = collections.Counter(k_cluster_not_normalized.labels_)

print("Cluster Member Counts\n")
print(f"Cluster-0 member count = {counts[0]}")
print(f"Cluster-1 member count = {counts[1]}")
print(f"Cluster-2 member count = {counts[2]}")

"""
 With Normalization enabled, before sending dataset for clustering we are getting following clusters
 
 Cluster Member Counts

 Cluster-0 member count = 61
 Cluster-1 member count = 50
 Cluster-2 member count = 39
 
 
 And without normalization, the clusters look something like following
 
 Cluster-0 member count = 62
 Cluster-1 member count = 50
 Cluster-2 member count = 38
"""
