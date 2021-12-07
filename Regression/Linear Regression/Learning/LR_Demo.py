import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets ,linear_model
from sklearn.metrics import mean_squared_error

#Importing datasets
diabetes = datasets.load_diabetes()
 
#it will print all the keys
# print(diabetes.keys())
# The list of keys
# ['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module'] 
# print(diabetes.data)
# print(diabetes.DESCR)

# diabetes_x = diabetes.data[:, np.newaxis,2] 
diabetes_x = np.array([[1],[2],[3]]) # While using this line remove the plot functions as it cant be plot
#creates an array of array which will be x axis
# print(diabetes_x)

#Partitioning x(FEATURES)
diabetes_x_train = diabetes_x # -30 = last 30 records
diabetes_x_test = diabetes_x # -30 = first 30 records

#Now while partitioning y make sure you take same number of records beacuse the data in x correspponds to y and any imbalance will lead to formation of missing values

#Partitioning y(LABELS)
diabetes_y_train = np.array([[3],[2],[4]])
diabetes_y_test = np.array([[3],[2],[4]])

#Calling the imp function
model = linear_model.LinearRegression() 

#Fitting the data 
model.fit(diabetes_x_train,diabetes_y_train)
# When you call fit method it estimates the best representative function for the the data points (could be a line, polynomial or discrete borders around). With that representation, you can calculate new data points.

#Predicting the outcome
diabetes_y_predicted=model.predict(diabetes_x_test)

#Printing the mean squared error
print("the mean squared error is: ", mean_squared_error(diabetes_y_test, diabetes_y_predicted))

#plotting the model
print("Weights: ", model.coef_)
print("Intercept: ", model.intercept_)

#scatter plot
plt.scatter(diabetes_x_test, diabetes_y_test)
plt.plot(diabetes_x_test, diabetes_y_predicted)
plt.show()
