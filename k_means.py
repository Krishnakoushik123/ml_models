import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score,classification_report
#deifing the feature in from the dataset
x,y=make_blobs(n_samples=1000,centers=3,cluster_std=0.60,random_state=42)
#divide the dataset into training and testing data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
#defining the model and train with the data
model= KMeans(n_clusters=3, init='k-means++', max_iter=300, random_state=42)
y_preds=model.fit_predict(x)
model.fit(x_train,y_train)
predicts=model.predict(x_test)
#evaluate the performance of the model with metrics
'''accuracy=accuracy_score(y_test,predicts)
print(f"accuracy of the model:{accuracy*100:2f}%")
print("classification_report")
print(f"{classification_report(y_test,predicts)}")'''

#visualizing the clusters using matplot
plt.figure(figsize=(8,6))
plt.scatter(x[y_preds==0,0],x[y_preds==0,1],s=50,c='blue',marker='o',edgecolors='black',label='cluster 1')
plt.scatter(x[y_preds==1,0],x[y_preds==1,0],s=50,c='blue',marker='s',edgecolors='black',label='cluster 2')
plt.scatter(x[y_preds==2,0],x[y_preds==2,1],s=50,c='blue',marker='v',edgecolors='black',label='cluster 3')
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], s=250, marker='*', c='red', edgecolors='black', label='Centroids')
plt.title("k means clustering in unsupervised learning")
plt.xlabel("feature 1")
plt.ylabel("feature 2")
plt.legend("lower right")
plt.grid(True)
plt.show()