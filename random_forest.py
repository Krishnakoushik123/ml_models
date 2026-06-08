import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score,classification_report
#defining and loading the dataset
data=load_wine()
x=data.data
y=data.target
#dividing the dataset into training and testing data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
#defining the random_forest model and training with data
model=RandomForestClassifier(n_estimators=100,max_features='sqrt',random_state=42)
model.fit(x_train,y_train)
y_preds=model.predict(x_test)
#evaluating the model performance
accuracy=accuracy_score(y_test,y_preds)
print(f"accuracy of the model:{accuracy*100:2f}%\n")
print("classification report of the random forest model")
print(f"{classification_report(y_test,y_preds,target_names=data.target_names)}")
#vizualizing the 1  tree from random forest
single_tree=model.estimators_[0]
plt.figure(figsize=(20,10))
plot_tree(
    single_tree,
    feature_names=data.feature_names,  # Shows actual names like 'alcohol', 'magnesium'
    class_names=data.target_names,    # Shows the wine class names
    filled=True,                      # Colors the nodes based on the majority class
    rounded=True,                     # Makes the node boxes look pretty with rounded corners
    max_depth=3                       # Limits the visual depth so it doesn't get cluttered
)
plt.title("single decision tree of a random forest")
plt.show()