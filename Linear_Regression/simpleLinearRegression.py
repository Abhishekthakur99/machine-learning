#importing libraries....
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt`
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


#loading data 
dataFrame = pd.read_csv('ex1data1.txt')
#print(dataFrame)

#chosing Features and output

X = dataFrame.iloc[:, :-1].values
y = dataFrame.iloc[:, 1].values

#plotting data

plt.scatter(X,y,color='red',marker='x')
plt.title('Popualation in 10,000s v/s Price in 10,000s $')
plt.xlabel('Popualation in 10,000s')
plt.ylabel(' Price in 10,000s $')
plt.show()

#spliting data into training and test datasets

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2,random_state=0)
#Trianing the model

regressor = LinearRegression()
regressor.fit(X_train, y_train)

#printing interceptor
print('intercept is',regressor.intercept_)

#printing coefficient
print('coefficient is',regressor.coef_)

#making predictions
y_pred = regressor.predict(X_test)
print('Actual',y_test)
print('Predicted',y_pred)

#printing Mean Squared Error

print('Mean Square Error',metrics.mean_squared_error(y_test,y_pred))

#visualizing training set results
plt.scatter(X_train, y_train,color='red',marker='x')
plt.plot(X_train,regressor.predict(X_train),color='black',label='LinearRegression')
plt.legend()
plt.title('Visualizing training set results')
plt.xlabel('Popualation in 10,000s')
plt.ylabel('Price in $ 10,000s')
plt.show()

#visualizing test set results
plt.scatter(X_test, y_test,color='red',marker='x')
plt.plot(X_train,regressor.predict(X_train),color='black',label='LinearRegression')
plt.legend()
plt.title('Visualizing test set results')
plt.xlabel('Popualation in 10,000s')
plt.ylabel('Price in $ 10,000s')
plt.show()