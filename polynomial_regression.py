import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error,r2_score
#creating random data using random function
np.random.seed(42)
x=6*np.random.rand(100,1)-3
y=4*x**2+x+np.random.rand(100,1)
#visualizing these random dataset
'''plt.scatter(x,y,color="blue",label="original data",alpha=0.7)
plt.title("random dataset")
plt.xlabel("features")
plt.ylabel("y values")
plt.legend()
plt.show()'''
#loading and training the polynomial features
poly_features=PolynomialFeatures(degree=2,include_bias=False)
poly_x=poly_features.fit_transform(x)
print(f"original x_dataset first row:{x[0]}")
print(f"first row of poly_x values:{poly_x[0]}")
#linear regression line with the multiple x_features
model=LinearRegression()
model.fit(poly_x,y)
#collecting the random data for predicting the y_values
x_new=np.linspace(3,-3,100).reshape(100,1)
x_new_poly=poly_features.transform(x_new)
y_new=model.predict(x_new_poly)
#predicting the actual y values
y_pred=model.predict(poly_x)
#model evaluation
print(f"model intercept:{model.intercept_[0]:2f}")
print(f"model coeficent:{model.coef_[0]}")
print(f"mean_squared_error:{mean_squared_error(y,y_pred):2f}")
print(f"r2_score:{r2_score(y,y_pred):2f}")
#visualizing the datapoints with linear regression line 
plt.figure(figsize=(8,6))
plt.scatter(x,y,color="blue",label="data points",alpha=0.7)
plt.plot(x_new,y_new,color="red",label="linear model")
plt.title("polynomial regression")
plt.xlabel("x values")
plt.ylabel("predictions")
plt.legend()
plt.show()