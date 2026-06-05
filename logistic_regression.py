import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn .model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
#defining the features and required data 
x,y=make_classification(n_samples=10000,n_features=2,n_redundant=0,n_clusters_per_class=1,random_state=42)
#dividing the original data into training and testing data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
#training the data on logistic regression
model=LogisticRegression()
model.fit(x_train,y_train)
#predict the output and classify it to a cluster
y_pred=model.predict(x_test)
y_probability=model.predict_proba(x_train)
#evaluating the model with matrics
accuracy=accuracy_score(y_test,y_pred)
confusion_matrics1=confusion_matrix(y_test,y_pred)
print(f"Accuracy_score{accuracy*100}%\n")
print("confusion matrics")
print(confusion_matrics1)
print(classification_report(y_test,y_pred))

#model selected weight and bias
print(f"weight (w):{model.coef_}")
print(f"bias (b):{model.intercept_}")
