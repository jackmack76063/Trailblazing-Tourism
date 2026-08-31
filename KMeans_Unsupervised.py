###
# Name: Mackenzie Jackson
# Module: 7
# "Project: Unsupervised Learning Using KMeans"
# Date: 11/11/2025
###
 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv("AllTrails_kmeans.csv")

###Visualization Before Clustering###
#fig = plt.figure(figsize=(10,12))
#ax = fig.add_subplot(111, projection='3d')

#ax.scatter(df["HardPercent"], df["AvgLen_mi"], df["ReviewCount"], alpha = 0.8, s = 50)
#ax.set_xlabel("Hard Percent")
#ax.set_ylabel("Average Trail Length (mi)")
#ax.set_zlabel("Review Count")
#ax.set_title("Trail Difficulty, Length, and Popularity (Before Clustering)")
#plt.tight_layout()
#plt.show()

###KMeans Clustering###
k_cols = df[['HardPercent', 'AvgLen_mi', 'ReviewCount']]
MyKMeans = KMeans(n_clusters=4)
KMeansResults = MyKMeans.fit(k_cols)

print("\nCluster assignments (labels):")
print(KMeansResults.labels_)

print("\nCluster centroids:")
print(KMeansResults.cluster_centers_)

#Prediction
#difficult of 0.5, avg length of 8 miles, review count of 50000 
k_predict = MyKMeans.predict([[0.5, 8, 50000]])
print(f"\nPredicted cluster [0.5, 8, 50000]: {k_predict}\n")

###Visualization After Clustering###
fig = plt.figure(figsize=(10,12))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df["HardPercent"], df["AvgLen_mi"], df["ReviewCount"], c=KMeansResults.labels_, cmap="rainbow", s=50, alpha=0.8)

centroids = KMeansResults.cluster_centers_
ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 1], s = 600, c = 'black', edgecolor='white', marker='*', depthshade=False)


ax.set_xlabel("Hard Percent")
ax.set_ylabel("Average Trail Length (mi)")
ax.set_zlabel("Review Count")
ax.set_title("Trail Difficulty, Length, and Popularity (Before Clustering)")
plt.tight_layout()
plt.show()