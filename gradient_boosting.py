import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score,classification_report
#defining the required features and labels from the dataset
x,y=make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=8,
    random_state=42
)
#dividing the dataset into training and testing data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
#defining the model and traing with dataset
model=GradientBoostingClassifier(n_estimators=100,max_depth=3,learning_rate=0.1,random_state=42)
model.fit(x_train,y_train)
y_preds=model.predict(x_test)
#evaluating the model performace by calculating the following metrics
accuracy=accuracy_score(y_test,y_preds)
print(f"accuracy of the model:{accuracy*100:2f}%")
print("classification report")
print(f"{classification_report(y_test,y_preds)}")