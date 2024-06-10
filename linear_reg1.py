import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np

dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_predtrain = regressor.predict(x_train)
y_predtest = regressor.predict(x_test)

plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, y_predtrain, color='blue')
plt.title('Salary vs experience (training set)')
plt.xlabel('yrs of experience')
plt.ylabel('salary')
plt.show

plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, y_predtrain, color='blue')
plt.title('Salary vs experience (test set)')
plt.xlabel('yrs of experience')
plt.ylabel('salary')
plt.show


