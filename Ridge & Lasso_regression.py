import numpy as np
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
#defining the required features and data in the dataset
x,y=make_regression(n_samples=100,n_features=20,n_informative=5,noise=25.0,random_state=42)
#dividing the dataset into training and testing data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
##  CRITICAL STEP: Feature Scaling
# Regularization penalizes coefficient size, so all features must be on the same scale.
scalar=StandardScaler()
x_train_scale=scalar.fit_transform(x_train)
x_test_scale=scalar.transform(x_test)
#defining the models and training the models with the data
model_lr=LinearRegression()
model_ridge=Ridge(alpha=10.0)
model_lasso=Lasso(alpha=5.0)

model_lr.fit(x_train_scale,y_train)
model_ridge.fit(x_train_scale,y_train)
model_lasso.fit(x_train_scale,y_train)
#predicting the output y values
pred_lr=model_lr.predict(x_test_scale)
pred_ridge=model_ridge.predict(x_test_scale)
pred_lasso=model_lasso.predict(x_test_scale)
#evaluating the error with MSE
print(f"model_linear_regression:{mean_squared_error(y_test,pred_lr)}")
print(f"model_ridge:{mean_squared_error(y_test,pred_ridge)}")
print(f"model_lasso:{mean_squared_error(y_test,pred_lasso)}")

#Observe the Coefficient Shrinkage (Feature Selection in Lasso)
print(f"Standard LR:{np.sum(model_lr.coef_==0)}")
print(f"ridge:{np.sum(model_ridge.coef_==0)}(Shrinks them, but rarely to exactly zero)")
print(f"lasso:{np.sum(model_lasso.coef_==0)}(Acts as an automatic feature selector)")