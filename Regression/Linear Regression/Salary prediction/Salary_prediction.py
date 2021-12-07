
import pandas as pd
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt

#___________Reading the test and train datasets________
train_data = pd.read_csv('E:\\DATA SETS\\Simple regression data\\train.csv')
train = train_data.dropna() #__Training data_____

#Dropna() will remove any anomlies if present

test_data = pd.read_csv('E:\\DATA SETS\\Simple regression data\\test.csv')
test = test_data.dropna() #__Testing data_____

#___________Creating the variables _____
x_train = np.array(train.iloc[:,:-1].values)  #Independant variable (Experience)
y_train =np.array(train.iloc[:,1].values)    #Dependant variable (Salary)

x_test = np.array(test.iloc[:,:-1].values)  #Independant variable (Experience)
y_test =np.array(test.iloc[:,1].values)    #Dependant variable (Salary)

# print(x_test)
# print(y_test)

#________Training the model______

model = linear_model.LinearRegression()

model.fit(x_train, y_train)

#_______printing the y_test values__
print("______________actual values_______-")
print(y_test)
#_______Predicting from the model____
Prediction = model.predict(x_test)
print("______Predicted values__________")
print(Prediction)
#We will provide it an independant variable and it will print the dependant ones

#_________metrics____
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, Prediction)
print("__________MEAN_absolute_error_________")
print(mae)

#_________Intecept_____
print("_______Intercept_______")
print( model.intercept_)

#___Plotting the graph
plt.plot(Prediction, x_test)
plt.scatter(y_test, x_test)
plt.show()

#_______calculating the value of m
# y = mx+c
#considering a point on the line from the given data
x = 8.2 
y = 104044.71643394
c = 25011.348860873994
m = (y-c)/x
print("_______value of slope(m)_____")
print(m)
#9638.215557690977


