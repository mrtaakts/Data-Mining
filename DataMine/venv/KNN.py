import sqlite3
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

conn= sqlite3.connect('dataset.sqlite')

df= pd.read_sql_query("SELECT * FROM Iris", conn)

print(df.head()) # iris db'sindeki ilk 5 veriyi gösteriyorum

y=df["Species"]
df = df.dropna()
X=df.drop(["Species"], axis=1)
print(y)
print(X)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.94, shuffle=True)

knn = KNeighborsClassifier()
knn_model = knn.fit(X_train, y_train)
knn_model
print(knn_model)

y_pred = knn_model.predict(X_test)

print(accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))

