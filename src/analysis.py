# author: Aakanksha Dimri
# date: 2020-01-22

'''This script fits a model and outputs several images and csv reports

    Usage: src/analysis.py --input1=<input1> --input2=<input2> --output=<output>

    Options:
    --input1=<input1>  Path (including filename) to training data (which needs to be saved as a csv file).
    --input2=<input2>  Path (including filename) to testing data (which needs to be saved as a csv file).
    --output=<output>  Name of directory to be saved in, no slashes nesscesary, 'results' folder recommended.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, plot_confusion_matrix, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.metrics import recall_score, precision_score
from sklearn.model_selection import GridSearchCV
from docopt import docopt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn import tree 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
import time
import autotime

opt = docopt(__doc__)

# Code attribution for function get_scores: DSCI 571, Lecture 8
def get_scores(model, 
                X_train, y_train,
                X_test, y_test
               ):
    """
    Returns train and test accuracy given a model
    train and test X and y portions
    Parameters:
    ----------
    model: sklearn classifier model
        The sklearn model
    X_train: numpy.ndarray        
        The X part of the train set
    y_train: numpy.ndarray
        The y part of the train set    
    X_test: numpy.ndarray        
        The X part of the test set
    y_valid: numpy.ndarray
        The y part of the test set    
    Returns
    -------
        train_score: float
        test_score: float
            
    """              
    return (model.score(X_train, y_train)), (model.score(X_test, y_test))


def main (input1,input2,output):

    # Read wrangled csv files
    df_train = pd.read_csv(input1)
    df_test = pd.read_csv(input2)
    X_train = df_train.drop(['Approved'], 1)
    y_train = df_train[['Approved']]
    X_test = df_test.drop(['Approved'], 1)
    y_test = df_test[['Approved']]

## Encoding categorical variables
    categorical_features = ['Sex','Ethnicity', 'Married','BankCustomer','EducationLevel','PriorDefault','Employed','DriversLicense','Citizen','ZipCode']
    preprocessor = ColumnTransformer(
    transformers=[('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False), categorical_features)])

    X_train = pd.DataFrame(preprocessor.fit_transform(X_train))
    X_test = pd.DataFrame(preprocessor.transform(X_test))
    y_train = y_train.to_numpy().ravel()
    y_test = y_test.to_numpy().ravel()

    
    ## Running models with default parameters
    # empty dictionary to store results
    results_dict = {}
    models = {
          'random forest' : RandomForestClassifier(random_state=23), 
          'xgboost' : XGBClassifier(random_state=23),
          'lgbm': LGBMClassifier(random_state=23)
         }

    for model_name, model in models.items():
        t = time.time()
        #print(model_name, ":")
        clf = Pipeline(steps=[('classifier', model)])
        clf.fit(X_train, y_train);
        train_score, test_score = get_scores(clf, X_train, y_train, 
                                       X_test, y_test)
        elapsed_time = time.time() - t
        results_dict[model_name] = [round(train_score,3), round(test_score,3), round(elapsed_time,4)]
        
    ## Storing results of default model settings
    model_compare_dataframe = pd.DataFrame(results_dict)
    model_compare_dataframe.to_csv(output + "model_compare.csv")

    
    ### Hyper parameter optimisation for chosen model ie. Random Forest
    hyper_parameters = [
        {
        'n_estimators': [3, 5, 10, 50, 100],
        'criterion': ['gini', 'entropy'], 
        'max_depth': [10, 20, 50, None]
        }
    ]
    
    clf = GridSearchCV(
    RandomForestClassifier(random_state=23),
        hyper_parameters,
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=23),
        verbose=0
        )
    best_model = clf.fit(X_train, y_train)
    

    # Measure accuracies on test and train data
    train_predictions = best_model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_predictions)
    test_predictions = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    test_recall = recall_score(y_test, test_predictions)
    test_precision = precision_score(y_test, test_predictions)
    auc_score = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
    accuracies_df = pd.DataFrame(index=['test accuracy', 'test recall', 'auc score'], data={
        'result': [test_accuracy, test_recall, auc_score]
    })
    accuracies_df.to_csv(output + "accuracy_report.csv")

    # plot and report confusion matrix for test data
    plot_confusion_matrix(best_model, X_test, y_test)
    report = classification_report(y_test, test_predictions, output_dict=True)
    report_df = pd.DataFrame(report)
    report_df.to_csv(output + "classification_report.csv")
    

    # compute and save roc curve for test data
    fpr, tpr, thresholds = roc_curve(
        y_test, best_model.predict_proba(X_test)[:, 1])
    # plt.plot(fpr, tpr)
    plt.title('ROC report')
    # plt.plot((0, 1), (0, 1), '--k')
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.savefig(f'./{output}/roc.png')

if __name__ == "__main__":
    main(input1=opt["--input1"],input2=opt["--input2"], output=opt["--output"])