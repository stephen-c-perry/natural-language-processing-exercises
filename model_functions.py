import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split



'''
This module holds functions for generating dataframes of model recall results for performance comparison
'''


def split_data(df, strat= None):
    '''
    take in a DataFrame and return train, validate, and test DataFrames.
    return train, validate, test DataFrames.
    '''
    train_validate, test = train_test_split(df, test_size=.16, random_state=123, stratify=df[strat])
    train, validate = train_test_split(train_validate,
                                       test_size=.2, 
                                       random_state=123, 
                                       stratify=train_validate[strat])
    return train, validate, test



#function to run multiple Decision Tree models, calculate recall and difference of recall in train and val sets

def get_decision_tree_multiple(X_Train, y_Train, X_val, y_val):
    metrics = []

    #create many decision tree models trying different hyperparameters and get the best performance
    for j in range (1, 10):
        for i in range(2, 10):
            clf = DecisionTreeClassifier(max_depth=i, min_samples_leaf=j, random_state=123)

            clf = clf.fit(X_Train, y_Train)
                        
            # make predictions on train set
            y_pred_train = clf.predict(X_Train)
            # make predicitons on val set
            y_pred_val = clf.predict(X_val)
           
            #calculate recall
            in_sample_recall = recall_score(y_Train, y_pred_train, pos_label=1)
            out_of_sample_recall = recall_score(y_val, y_pred_val, pos_label=1)

            output = {
                "min_samples_per_leaf": j,
                "max_depth": i,
                "train_recall": in_sample_recall,
                "validate_recall": out_of_sample_recall,
            }

            metrics.append(output)
            

    df1 = pd.DataFrame(metrics)
    #make new column to calculate difference in accuracy between train and val
    df1["difference"] = df1['train_recall'] - df1['validate_recall']
    df1_sorted = df1.sort_values(by=['validate_recall'], ascending=False).head(10)
    

    return df1_sorted




#function to run multiple random forest models, calculate recall and difference of recall in train and val sets

def get_random_forest_multiple(X_Train, y_Train, X_val, y_val):
    metrics = []

    #create many decision tree models trying different hyperparameters and get the best performance
    for j in range (1, 10):
        for i in range(2, 10):
            clf = RandomForestClassifier(max_depth=i, min_samples_leaf=j, random_state=123)

            clf = clf.fit(X_Train, y_Train)
                        
            # make predictions on train set
            y_pred_train = clf.predict(X_Train)
            # make predicitons on val set
            y_pred_val = clf.predict(X_val)
           
            #calculate recall
            in_sample_recall = recall_score(y_Train, y_pred_train, pos_label=1)
            out_of_sample_recall = recall_score(y_val, y_pred_val, pos_label=1)

            output = {
                "min_samples_per_leaf": j,
                "max_depth": i,
                "train_recall": in_sample_recall,
                "validate_recall": out_of_sample_recall,
            }

            metrics.append(output)
            

    df1 = pd.DataFrame(metrics)
    #make new column to calculate difference in accuracy between train and val
    df1["difference"] = df1['train_recall'] - df1['validate_recall']
    df1_sorted = df1.sort_values(by=['validate_recall'], ascending=False).head(10)
    

    return df1_sorted




#function to run multiple random forest models, calculate recall and difference of recall in train and val sets

def get_knn_multiple(X_Train, y_Train, X_val, y_val):
    metrics = []

    #create many decision tree models trying different hyperparameters and get the best performance
    for j in range (1, 10):
        for i in range(2, 10):
            #choose KNN algo
            clf = KNeighborsClassifier(n_neighbors=i, weights='uniform')
            #fit algo to model
            clf = clf.fit(X_Train, y_Train)
                        
            # make predictions on train set
            y_pred_train = clf.predict(X_Train)
            # make predicitons on val set
            y_pred_val = clf.predict(X_val)
           
            #calculate recall
            in_sample_recall = recall_score(y_Train, y_pred_train, pos_label=1)
            out_of_sample_recall = recall_score(y_val, y_pred_val, pos_label=1)

            output = {
                "min_samples_per_leaf": j,
                "max_depth": i,
                "train_recall": in_sample_recall,
                "validate_recall": out_of_sample_recall,
            }

            metrics.append(output)
            

    df1 = pd.DataFrame(metrics)
    #make new column to calculate difference in accuracy between train and val
    df1["difference"] = df1['train_recall'] - df1['validate_recall']
    df1_sorted = df1.sort_values(by=['validate_recall'], ascending=False).head(10)
    

    return df1_sorted



#isolating target variable in train, validate, test sets
def isolate_target(train, validate, test, target):
    '''
    Seperates target variable from original df for modeling
    '''
    X_Train = train.drop(columns = [target])
    y_Train = train[target]

    X_val = validate.drop(columns = [target])
    y_val = validate[target]

    X_test = test.drop(columns = [target])
    y_test = test[target]
    return X_Train, y_Train, X_val, y_val, X_test, y_test



#get dummies and drop any columns
def df_classification_ready(df, cols= None):
    '''
    prepares df for classification by getting dummies and dropping columns
    '''
    df = pd.get_dummies(df)
    df = df.drop(columns= [cols])

    return df


def test_predictions(X_Test, y_test):
    clf = RandomForestClassifier(max_depth=7, min_samples_leaf=5, random_state=123)

    clf = clf.fit(X_Test, y_test)
                        
    # make predictions on test set
    y_pred_test = clf.predict(X_Test)
    
    in_sample_recall = recall_score(y_test, y_pred_test, pos_label=1)
    print(f'Recall value of test predicitions: {round(in_sample_recall,4)}')

    test_results = pd.DataFrame(y_pred_test)

    return test_results