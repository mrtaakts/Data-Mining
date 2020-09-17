import sqlite3
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
conn= sqlite3.connect('data.sqlite')

df= pd.read_sql_query("SELECT * FROM fish_data", conn)
y=df["Spec"]
df = df.dropna()
X=df.drop(["Spec"], axis=1)
scaler = MinMaxScaler(feature_range=(0,1))
X=scaler.fit_transform(np.array(X))
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30, shuffle=True)
decision = DecisionTreeClassifier(criterion='entropy',
                                  splitter='best',
                                  min_samples_split=2,
                                  min_samples_leaf=2,
                                  min_weight_fraction_leaf=0.0,
                                  max_features=6)
decision_model = decision.fit(X_train, y_train)
decision_model
y_pred = decision_model.predict(X_test) # nemo
print(accuracy_score(y_test, y_pred))
confusion_matrix =confusion_matrix(y_test, y_pred)
print(confusion_matrix)
def counts_from_confusion(confusion):
    """
    Obtain TP, FN FP, and TN for each class in the confusion matrix
    """

    counts_list = []

    # Iterate through classes and store the counts
    for i in range(confusion.shape[0]):
        tp = confusion[i, i]

        fn_mask = np.zeros(confusion.shape)
        fn_mask[i, :] = 1
        fn_mask[i, i] = 0
        fn = np.sum(np.multiply(confusion, fn_mask))

        fp_mask = np.zeros(confusion.shape)
        fp_mask[:, i] = 1
        fp_mask[i, i] = 0
        fp = np.sum(np.multiply(confusion, fp_mask))

        tn_mask = 1 - (fn_mask + fp_mask)
        tn_mask[i, i] = 0
        tn = np.sum(np.multiply(confusion, tn_mask))

        counts_list.append({'Class': i,
                            'acc': (tp + tn)/(tp + tn + fp + fn),
                            'sensitivity':tp/(tp+fn),
                            'specificity':tn/(tn+fp)

                            })

    return counts_list
list = counts_from_confusion(confusion_matrix)
print(list)
plot_confusion_matrix(decision,X_test,y_test)
plt.show()