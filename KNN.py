import numpy as np
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#define the required data from the dataset
#x,y=make_blobs(n_samples=100,centers=2,random_state=42,cluster_std=1.5)
x=np.array([
    [10,3],[12,4],[8,3],   #(action:high,comedy:low)
    [1,9],[3,12],[2,10]    #(action:low,comedy:high)
])
y=np.array([0,0,0,1,1,1])
#dividing the dataset into training and testing datasets
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
#defining the model and training the data
knn_model=KNeighborsClassifier(n_neighbors=1)
knn_model.fit(x_train,y_train)
y_preds=knn_model.predict(x_test)
print("sample classified into:",y_preds)
#evaluating the model with accuracy asore
accuracy=accuracy_score(y_test,y_preds)
print(f"accuracy of the model:{accuracy*100:2f}%\n")
#predicting with sample data
test_sample=np.array([[2,13]])
y_preds1=knn_model.predict(test_sample)
print("sample classified into:",y_preds1)