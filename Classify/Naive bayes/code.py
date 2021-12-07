import pandas as pd
dataset = pd.read_csv("E:\DATA SETS\covid_classify.csv")

X= dataset.iloc[:,0:10]
Y= dataset.iloc[:,-1:]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.4)

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train,Y_train)
y_pred = model.predict(X_test)

print("_______original values____")
print(Y_test)
print("Predicted values")
print(y_pred)

from sklearn.metrics import confusion_matrix, f1_score,accuracy_score
accuracy = accuracy_score(Y_test,y_pred)
print("________Accuracy______")
print(accuracy)
cm = confusion_matrix(Y_test, y_pred)
print("_____Confusion Matrix_____")
print(cm)