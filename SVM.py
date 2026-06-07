import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,classification_report
#define the dataset along with required number of features
x,y=make_classification(n_samples=10000,n_features=100,n_informative=15,random_state=42,n_classes=2)
#dividing the dataset into training and testing data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print("training dataset shape:",x_train.shape)
print("testing dataset shape:",x_test.shape)
#defining the svc model and traing with tha data
svc_model=SVC(kernel='rbf',C=1.0,gamma='scale',random_state=42)
svc_model.fit(x_train,y_train)
y_preds=svc_model.predict(x_test)
#evaluating the model performnce by using the metrics
accuracy=accuracy_score(y_test,y_preds)
print(f"accuracy of the model:{accuracy*100:2f}%")
print("classification_report")
print(f"{classification_report(y_test,y_preds)}")
#evaluates the number of support vectors in a single class
print(f"number of support vectors per class:{svc_model.n_support_}")

