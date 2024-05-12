########################################################################################################################
# IMPORTS
########################################################################################################################

import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

########################################################################################################################
# PART 1 // IMPLEMENT k-Means
########################################################################################################################

'''
In this exercise we want to implement the k-means algorithm
you'll be provided with a set of initial means and you should:
 - (1) assign data points based on euclidean distance to the clusters
 - (2) calculate new centers
    ... interate

data:
-----
- https://www.kaggle.com/aryashah2k/credit-card-customer-data?select=Credit+Card+Customer+Data.csv
- credit card customer data
- we will preselect two features for testing k-means, but be encouraged to try the algorithm on your won

what you don't need to care about:
---------------------------------
- missing values
- categorical values (which are problematic for kmeans anyway)

visualize the cluster updates:
-----------------------------
- show how cluster centers moves over multiple iteration
'''

# load data #
# -- pre-coded -- #
pth = r"ccc.csv"
df = pd.read_csv(pth)
df.describe()
df.dtypes
df.isna().sum()
df.drop(["Sl_No", "Customer Key"], axis=1, inplace=True)  # drop keys
df = df.astype("float64")  # we will need float values for k-means
df.dtypes

# look at the data (pair plots) #
# select only a few meaningful attributes #
# checkout: https://seaborn.pydata.org/generated/seaborn.jointplot.html #
# -- pre coded -- #
sns.pairplot(df)
df = df[
    ["Total_visits_online", "Avg_Credit_Limit"]]  # we choose these two features, it seems there are two clusters here
sns.jointplot(data=df, x="Total_visits_online", y="Avg_Credit_Limit", kind="kde")
sns.jointplot(data=df, x="Total_visits_online", y="Avg_Credit_Limit", kind="hist")

# implement k-means #
m = np.array([[2.5, 100000], [10.0, 125000]])  # we use two cluster centers

# scale the data first and afterwards m ! (So that that m does not jump to extreme values)
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)
m_scaled = scaler.transform(m)

# iterate
for i in range(10):
    # k-means #1 ... distance and cluster assignment
    distances = np.zeros((df_scaled.shape[0], m_scaled.shape[0]))
    clusters = np.zeros(df_scaled.shape[0])

    for j in range(df_scaled.shape[0]):  # for each data point
        for k in range(m_scaled.shape[0]):  # for each cluster
            distances[j, k] = np.linalg.norm(df_scaled[j] - m_scaled[k])  # euclidean distance
        clusters[j] = np.argmin(distances[j])  # assign to cluster

    # k-means #2 ... means update
    for j in range(m_scaled.shape[0]): # for each cluster
        m_scaled[j] = np.mean(df_scaled[clusters == j], axis=0)  # update cluster center

    # visualize results
    plt.figure(figsize=(10, 6))
    colors = ['red', 'blue']
    for j in range(m_scaled.shape[0]): # for each cluster
        cluster_data = df_scaled[clusters == j]  # get data points of cluster j
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], c=colors[j], label=f'Cluster {j}')  # plot data points
    plt.scatter(m_scaled[:, 0], m_scaled[:, 1], c='black', s=300, label='Centroids', marker='*')  # plot centroids
    plt.xlabel('Total Visits Online (scaled)')
    plt.ylabel('Average Credit Limit (scaled)')
    plt.title('Visualization of Cluster Assignments and Centroids')
    plt.legend()
    plt.show()
