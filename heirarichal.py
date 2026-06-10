import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram,linkage
from sklearn.cluster import AgglomerativeClustering
#initializing the random dataset
x=np.array([
    [2,5],
    [1,5],
    [3,9],
    [2,8],
    [8,4],
    [7,2],
    [3,6],
    [9,1],
    [4,5]
])
link=linkage(x,method='ward')
#Plot the Dendrogram to decide the number of clusters
plt.figure(figsize=(8,6))
dendrogram(link,orientation='top',distance_sort='descending',show_leaf_counts=True)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Data Point Index")
plt.ylabel("Distance (Ward's)")
plt.show()

#defining the agglomerative model and train with data
model=AgglomerativeClustering(n_clusters=3,metrics='euclidean',linkage='ward')
y_preds=model.fit_predict(x)
#plot the graph of the agglomerative clusters
plt.figure(figsize=(10,8))
plt.scatter(x[:,0],x[:,1],c='cluster.label_',cmap='rainbow',s=100)
plt.title("Data Points Grouped into 3 Clusters")
plt.xlabel("X Feature 1")
plt.ylabel("x Feature 2")
plt.grid(True)
plt.show()