from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("normalized_DCL.csv")

data.head()
# print(data)
new_data = data.drop(labels=['Smell'], axis=1)

names = []
for name in new_data:
    names.append(name)

y_data = data['Smell'].as_matrix()
for i in range(len(y_data)):
    if y_data[i] == True:
        y_data[i] = 0
    else:
        y_data[i] = 1

y_data = pd.Series(y_data, name='Smell')

X=new_data
y=y_data


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training / 30% test

# print(X_train)
# print(y_train)
clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

# for i in range(len(y_pred)):
#     print(str(y_test.as_matrix()[i]) + " "+ str(y_pred[i]))

print("Precision/Recall/Fscore/Suport:",metrics.precision_recall_fscore_support(y_test.as_matrix(), y_pred))
print("Accuracy:",metrics.accuracy_score(y_test.as_matrix(), y_pred))
precision = metrics.precision_score(y_test.as_matrix(), y_pred)
recall = metrics.recall_score(y_test.as_matrix(), y_pred)
fmeasure = 2*(precision * recall) / (precision + recall)
print("F_measure:",fmeasure)
# print(("&&&&&&&&&&&&"))
# print(names)
feature_imp = pd.Series(clf.feature_importances_,index=names).sort_values(ascending=False)
feature_imp

# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()