import numpy as np
import pandas as pd
import matplotlib.pyplot as mtp
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
import math


data_set=pd.read_csv("Social_media_effect.csv")


data_set=data_set.drop("Which social media platform/s do you like the most or use the most?",axis=1)
print(data_set.isna().sum())


# Encode Categorical Columns
categ = ['How much time do you spend on social media in a day?','How much time do you spend on physical activities in a day?','Have you ever been a victim of any of these cyber crimes?','Which type of communication do you generally prefer?','Measurement Effect in human']
le = LabelEncoder()
data_set[categ] = data_set[categ].apply(le.fit_transform)

x=data_set.iloc[:,:-1].values
y=data_set.iloc[:, -1 ].values

imp=SimpleImputer(missing_values=np.nan,strategy="mean")
Imputer=imp.fit(x[:,1:2])
x[:,1:2]=imp.fit_transform(x[:,1:2])



print("\n\n\n\n")
print("\t\t\t\t\t\t\t Random Forest")

def evaluate_model(x, y):
    k_fold = KFold(10, shuffle=True ,random_state=0)

    predicted_targets = np.array([])
    actual_targets = np.array([])

    for train, test in k_fold.split(x):
        x_train, y_train, x_test, y_test = x[train], y[train], x[test], y[test]

        # Fit the classifier
        classifier =RandomForestClassifier(n_estimators= 10).fit(x_train, y_train)

        # Predict the labels of the test set samples
        predicted_labels = classifier.predict(x_test)

        predicted_targets = np.append(predicted_targets, predicted_labels)
        actual_targets = np.append(actual_targets, y_test)

    return predicted_targets, actual_targets,x_test,y_test,x_train,y_train,classifier

predicted_target, actual_target, x_test,y_test,x_train,y_train,classifier = evaluate_model(x, y)


print("Confusion Matrix: \n", metrics.confusion_matrix(actual_target,predicted_target ))
print("Accuracy: ", metrics.accuracy_score(actual_target,predicted_target))
print(metrics.classification_report(actual_target,predicted_target))


print("\n\nMean Absolute Error")
error = mae(actual_target,predicted_target)
print(error*100)
print("\n\n")

print("Root Mean Squred Error")
error = mse(actual_target,predicted_target)
rmse=math.sqrt(error)
print(rmse*100)
print("\n\n")

print("Relative Absolute Error")
def rae(actual_target,predicted_target):
  numerator=np.sum(np.abs(predicted_target - actual_target))
  denominator=np.sum(np.abs(np.mean(actual_target) - actual_target))
  return numerator/denominator

print(rae(actual_target,predicted_target)*100)
print("\n\n")


y_pred_proba=classifier.predict_proba(x_test)[:,1]
AUCLR=metrics.roc_auc_score(y_test, y_pred_proba)
print("AUC Score: ", AUCLR)
fprLR, tprLR, thresholdvalues= metrics.roc_curve(y_test, y_pred_proba)
mtp.plot([0,1], [0,1], color='red', linestyle='--')
mtp.plot(fprLR, tprLR, label="Random Forest (area= " +str(AUCLR)+")")
mtp.xlabel("False Positive Rate")
mtp.ylabel("True Positive Rate")
mtp.legend(loc="lower right")
mtp.show()







print("\n\n\n\n")
print("\t\t\t\t\t\t\t Adaboost")

def evaluate_model(x, y):
    k_fold = KFold(10, shuffle=True, random_state=0)

    predicted_targets = np.array([])
    actual_targets = np.array([])

    for train, test in k_fold.split(x):
        x_train, y_train, x_test, y_test = x[train], y[train], x[test], y[test]

        # Fit the classifier
        classifier =AdaBoostClassifier(n_estimators=100).fit(x_train, y_train)

        # Predict the labels of the test set samples
        predicted_labels = classifier.predict(x_test)

        predicted_targets = np.append(predicted_targets, predicted_labels)
        actual_targets = np.append(actual_targets, y_test)

    return predicted_targets, actual_targets,x_test, y_test,x_train,y_train,classifier

predicted_target, actual_target, x_test,y_test,x_train,y_train,classifier = evaluate_model(x, y)


print("Confusion Matrix: \n", metrics.confusion_matrix(actual_target,predicted_target ))
print("Accuracy: ", metrics.accuracy_score(actual_target,predicted_target))
print(metrics.classification_report(actual_target,predicted_target))



print("\n\nMean Absolute Error")
error = mae(actual_target,predicted_target)
print(error*100)
print("\n\n")

print("Root Mean Squred Error")
error = mse(actual_target,predicted_target)
rmse=math.sqrt(error)
print(rmse*100)
print("\n\n")



print("Relative Absolute Error")
def rae(actual_target,predicted_target):
  numerator=np.sum(np.abs(predicted_target - actual_target))
  denominator=np.sum(np.abs(np.mean(actual_target) - actual_target))
  return numerator/denominator

print(rae(actual_target,predicted_target)*100)
print("\n\n")


y_pred_proba=classifier.predict_proba(x_test)[:,1]
AUCLR=metrics.roc_auc_score(y_test, y_pred_proba)
print("AUC Score: ", AUCLR)
fprLR, tprLR, thresholdvalues= metrics.roc_curve(y_test, y_pred_proba)
mtp.plot([0,1], [0,1], color='red', linestyle='--')
mtp.plot(fprLR, tprLR, label="Adaboost (area= " +str(AUCLR)+")")
mtp.xlabel("False Positive Rate")
mtp.ylabel("True Positive Rate")
mtp.legend(loc="lower right")
mtp.show()






print("\n\n\n\n")
print("\t\t\t\t\t\t\t Gradient Boosting")

def evaluate_model(x, y):
    k_fold = KFold(10, shuffle=True, random_state=0)

    predicted_targets = np.array([])
    actual_targets = np.array([])

    for train, test in k_fold.split(x):
        x_train, y_train, x_test, y_test = x[train], y[train], x[test], y[test]

        # Fit the classifier
        classifier =GradientBoostingClassifier(n_estimators=100).fit(x_train, y_train)

        # Predict the labels of the test set samples
        predicted_labels = classifier.predict(x_test)

        predicted_targets = np.append(predicted_targets, predicted_labels)
        actual_targets = np.append(actual_targets, y_test)

    return predicted_targets, actual_targets, x_test, y_test,x_train,y_train,classifier

predicted_target, actual_target, x_test,y_test,x_train,y_train,classifier = evaluate_model(x, y)

print("Confusion Matrix: \n", metrics.confusion_matrix(actual_target,predicted_target ))
print("Accuracy: ", metrics.accuracy_score(actual_target,predicted_target))
print(metrics.classification_report(actual_target,predicted_target))


print("\n\nMean Absolute Error")
error = mae(actual_target,predicted_target)
print(error*100)
print("\n\n")

print("Root Mean Squred Error")
error = mse(actual_target,predicted_target)
rmse=math.sqrt(error)
print(rmse*100)
print("\n\n")

print("Relative Absolute Error")
def rae(actual_target,predicted_target):
  numerator=np.sum(np.abs(predicted_target - actual_target))
  denominator=np.sum(np.abs(np.mean(actual_target) - actual_target))
  return numerator/denominator

print(rae(actual_target,predicted_target)*100)
print("\n\n")



y_pred_proba=classifier.predict_proba(x_test)[:,1]
AUCLR=metrics.roc_auc_score(y_test, y_pred_proba)
print("AUC Score: ", AUCLR)
fprLR, tprLR, thresholdvalues= metrics.roc_curve(y_test, y_pred_proba)
mtp.plot([0,1], [0,1], color='red', linestyle='--')
mtp.plot(fprLR, tprLR, label="Gradient Boosting(area= " +str(AUCLR)+")")
mtp.xlabel("False Positive Rate")
mtp.ylabel("True Positive Rate")
mtp.legend(loc="lower right")
mtp.show()




print("\n\n\n\n")
print("\t\t\t\t\t\t\t MLP")

def evaluate_model(x, y):
    k_fold = KFold(10, shuffle=True, random_state=0)

    predicted_targets = np.array([])
    actual_targets = np.array([])

    for train, test in k_fold.split(x):
        x_train, y_train, x_test, y_test = x[train], y[train], x[test], y[test]

        # Fit the classifier
        classifier =MLPClassifier(random_state=0, max_iter=300).fit(x_train, y_train)

        # Predict the labels of the test set samples
        predicted_labels = classifier.predict(x_test)

        predicted_targets = np.append(predicted_targets, predicted_labels)
        actual_targets = np.append(actual_targets, y_test)

    return predicted_targets, actual_targets, x_test, y_test,x_train,y_train,classifier

predicted_target, actual_target, x_test,y_test,x_train,y_train,classifier = evaluate_model(x, y)


print("Confusion Matrix: \n", metrics.confusion_matrix(actual_target,predicted_target ))
print("Accuracy: ", metrics.accuracy_score(actual_target,predicted_target))
print(metrics.classification_report(actual_target,predicted_target))

print("\n\nMean Absolute Error")
error = mae(actual_target,predicted_target)
print(error*100)
print("\n\n")

print("Root Mean Squred Error")
error = mse(actual_target,predicted_target)
rmse=math.sqrt(error)
print(rmse*100)
print("\n\n")

print("Relative Absolute Error")
def rae(actual_target,predicted_target):
  numerator=np.sum(np.abs(predicted_target - actual_target))
  denominator=np.sum(np.abs(np.mean(actual_target) - actual_target))
  return numerator/denominator

print(rae(actual_target,predicted_target)*100)
print("\n\n")

y_pred_proba=classifier.predict_proba(x_test)[:,1]
AUCLR=metrics.roc_auc_score(y_test, y_pred_proba)
print("AUC Score: ", AUCLR)
fprLR, tprLR, thresholdvalues= metrics.roc_curve(y_test, y_pred_proba)
mtp.plot([0,1], [0,1], color='red', linestyle='--')
mtp.plot(fprLR, tprLR, label="MLP(area= " +str(AUCLR)+")")
mtp.xlabel("False Positive Rate")
mtp.ylabel("True Positive Rate")
mtp.legend(loc="lower right")
mtp.show()



print("\n\n\n\n")
print("\t\t\t\t\t\t\t Decision tree")


def evaluate_model(x, y):
    k_fold = KFold(10, shuffle=True,  random_state=0)

    predicted_targets = np.array([])
    actual_targets = np.array([])

    for train, test in k_fold.split(x):
        x_train, y_train, x_test, y_test = x[train], y[train], x[test], y[test]

        # Fit the classifier
        classifier =DecisionTreeClassifier(random_state=0).fit(x_train, y_train)

        # Predict the labels of the test set samples
        predicted_labels = classifier.predict(x_test)

        predicted_targets = np.append(predicted_targets, predicted_labels)
        actual_targets = np.append(actual_targets, y_test)

    return predicted_targets, actual_targets, x_test, y_test,x_train,y_train,classifier

predicted_target, actual_target, x_test,y_test,x_train,y_train,classifier = evaluate_model(x, y)


print("Confusion Matrix: \n", metrics.confusion_matrix(actual_target,predicted_target ))
print("Accuracy: ", metrics.accuracy_score(actual_target,predicted_target))
print(metrics.classification_report(actual_target,predicted_target))


print("\n\nMean Absolute Error")
error = mae(actual_target,predicted_target)
print(error*100)
print("\n\n")

print("Root Mean Squred Error")
error = mse(actual_target,predicted_target)
rmse=math.sqrt(error)
print(rmse*100)
print("\n\n")


print("Relative Absolute Error")
def rae(actual_target,predicted_target):
  numerator=np.sum(np.abs(predicted_target - actual_target))
  denominator=np.sum(np.abs(np.mean(actual_target) - actual_target))
  return numerator/denominator

print(rae(actual_target,predicted_target)*100)
print("\n\n")

y_pred_proba=classifier.predict_proba(x_test)[:,1]
AUCLR=metrics.roc_auc_score(y_test, y_pred_proba)
print("AUC Score: ", AUCLR)
fprLR, tprLR, thresholdvalues= metrics.roc_curve(y_test, y_pred_proba)
mtp.plot([0,1], [0,1], color='red', linestyle='--')
mtp.plot(fprLR, tprLR, label="Decision tree(area= " +str(AUCLR)+")")
mtp.xlabel("False Positive Rate")
mtp.ylabel("True Positive Rate")
mtp.legend(loc="lower right")
mtp.show()










