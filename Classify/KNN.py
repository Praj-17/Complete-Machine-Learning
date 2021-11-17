from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
#Loading datasets
iris = datasets.load_iris()

#Printing the Description
print(iris.DESCR)

#Creating features and labels
#These are Numpy Arrays by default
features = iris.data 
labels = iris.target
print(features, labels)

#Training the classifier
clf = KNeighborsClassifier()

#Fitting the data into classifier
clf.fit(features, labels)

#prdicting the output for the given set of sttributes
preds = clf.predict([[31,1,1,1]])
print(preds)







