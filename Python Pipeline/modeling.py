'''
Predict Income with Census Data
Rafael Guerra
April 2022

modeling.py
Performs Logistic Regression and Random Forest models and computes accuracy results

'''

# import libraries and classes
import data_ingestion
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

# Split data into X and y components
def splitXY(dataset):
    dataset = pd.read_csv(dataset)
    df = dataset.values
    X_df, y_df = df[:, :-1], df[:, -1]
    return X_df,y_df

# Perform Logistic Regression Model Analysis
def LogisticModel(X_train,y_train,X_test,y_test):

    # Fit Logistic Regression to data
    logReg = LogisticRegression(solver='liblinear',penalty='l2').fit(X_train, y_train)

    # Accuracy
    log_reg_accuracy = logReg.score(X_test, y_test)
    print('LOGISTIC REGRESSION Accuracy: ', log_reg_accuracy)

    # AUC
    log_reg_auc = metrics.roc_auc_score(y_test, logReg.predict_proba(X_test)[:, 1])
    print('LOGISTIC REGRESSION AUC: ',log_reg_auc)

    # Confusion Matrix
    predictions = logReg.predict(X_test)
    log_reg_cm = metrics.confusion_matrix(y_test, predictions)
    print('LOGISTIC REGRESSION Confusion Matrix: ', log_reg_cm)

# Perform Random Forest Model Analysis
def RandomForest(X_train,y_train,X_test,y_test):

    # Fit Random Forest to data
    rfm = RandomForestClassifier(max_depth=16, random_state=0)
    randomForest = rfm.fit(X_train, y_train)

    # Accuracy
    rf_accuracy = randomForest.score(X_test,y_test)
    print ('RANDOM FOREST Accuracy: ', rf_accuracy)

    # AUC
    rf_auc = metrics.roc_auc_score(y_test, randomForest.predict_proba(X_test)[:, 1])
    print ('RANDOM FOREST AUC: ', rf_auc)

    # Confusion matrix
    predictions = randomForest.predict(X_test)
    rf_cm = metrics.confusion_matrix(y_test, predictions)
    print('RANDOM FOREST Confusion Matrix: ', rf_cm)

if __name__ == "__main__":

    # Logistic Regression
    LogisticModel(splitXY('train_clean.csv')[0],splitXY('train_clean.csv')[1],splitXY('test_clean.csv')[0],splitXY('test_clean.csv')[1])

    # Random Forest
    RandomForest(splitXY('train_clean.csv')[0],splitXY('train_clean.csv')[1],splitXY('test_clean.csv')[0],splitXY('test_clean.csv')[1])