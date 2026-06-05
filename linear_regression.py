import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
#taking random values as data to the model
diabeties=datasets.load_diabetes()
df=pd.DataFrame(data=diabeties.data,columns=diabeties.feature_names)
df['target']=diabeties.target
print(df.head())
# Using the third feature for simplicity
x=df[['bmi']].values
y=df['target'].values
#dividing these data into training and testing data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
#defining the linear model from the sklearrn library
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
#evaluating the model performance with these matrics
print(f"mean_squared_error{mean_squared_error(y_test,y_pred):.2f}")
print(f"r2_score{r2_score(y_test,y_pred):.2f}")
#visualizing the original data and the linear regression line.
plt.figure(figsize=(8,6))
plt.scatter(x,y,color="blue",label="original_data")
sort_idx = np.argsort(x_test.ravel())
plt.plot(x_test[sort_idx], y_pred[sort_idx], color="red", linewidth=3, label="Regression Line")
plt.xlabel("Body Mass Index (BMI) - Scaled")
plt.ylabel("Disease Progression Score")
plt.title("Predicting Diabetes Progression using BMI")
plt.legend()
plt.grid(True,linestyle="--",alpha=0.5)
plt.show()