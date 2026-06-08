import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
#loading the dataset and defining the labels and features
iris=load_iris()
x=iris.data
y=iris.target
#spliting the dataset into training and testing data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
#defining the model and training with data
model=DecisionTreeClassifier(criterion='entropy',max_depth=3,random_state=42)
model.fit(x_train,y_train)
#evaluating the model performance
accuracy=model.score(x_test,y_test)
y_pred=model.predict(x_test)
print(f"model prediction:{y_pred}")
print(f"accuracy of the model:{accuracy*100}%\n")
#visualizing the decision tree
plt.figure(figsize=(12,8))
plot_tree(model,feature_names=iris.feature_names,class_names=iris.target_names,filled=True)
plt.show()