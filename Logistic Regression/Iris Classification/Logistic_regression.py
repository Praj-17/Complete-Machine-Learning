# Train a logistic Regression classifier to predict weahter a flower is iris virginica or not

from sklearn import datasets
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

#Impoerting datset
iris = datasets.load_iris()

#Playing with the datasets
# print(list(iris.keys()))
# # print(iris.DESCR)
# print(iris.data)
# print(iris.target)

x = iris.data[:,3:] #slicing it to get only the 3rd column
y = (iris.target == 2).astype(np.int)
# print(y,x)

#Training the Model
clf = LogisticRegression()
clf.fit(x,y)

#Predicting the output for given value 
example = clf.predict(([[3.1]]))
print(example)

#Plotting the visualization

#Taking 1000 values between 0, 3 and reshaping it to beacome a one day array 
x_new = np.linspace(0,3, 1000).reshape(-1,1)
# print(x_new)

#calculating the probabilites usin _proba function
y_prob = clf.predict_proba(x_new)

#It make the line colour green and label it as viginica
plt.plot(x_new,y_prob[:,1], "g-", label = "virginica" )
plt.show()
