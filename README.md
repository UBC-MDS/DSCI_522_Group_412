# DSCI_522_Group_412

# Proposal

## Motivation and Research question

Algorithms are used on an everyday basis to decide the outcome of credit applications. The purpose of our analysis is to answer the question,
"Given an individual's information, can we predict their credit approval status?" 

This would be a two step process:

* Exploring whether a correlation exists between features lik Age, Income, Credit Score, Debt levels etc. and the credit approval status?
* Can this relationship be used to predict if a person is granted credit?

## Description of the Data

The Credit approval dataset is taken from the archives of the machine learning repository of University of California, Irvine (UCI). A quick review of the dataset, shows that all of the values in the dataset have been converted to meaningless symbols to protect the confidentiality of the data. However, to make the analysis intuitive and easy to understand, we gave the variables working names based on the type of data. More information about the dataset can be found [here](http://archive.ics.uci.edu/ml/datasets/credit+approval).

## Analysis Plan

We plan to split our analysis in the following steps:

- For exploratory analysis we plan to generate several data visualizations to understand the underlying data; using automated tools like Pandas profiling
- Perform data transformations like appropriate handling of missing values, standardising numerical features, encoding categorical features etc. as needed
- Generate and apply the model or combination of model(s) to answer the research question:
  - We plan to analyse the dataset with classifiers like Decision Trees and Random Forests 
  - We also plan to explore ensembling techniques like combining Logistic regression and Classification algorithms to obtain higher accuracy
  - To further improve the accuracy of our model we plan to use techniques such as K-fold cross validation split, Grid search cross validation etc.

## Results

To communicate the results effectively we plan on using Recall, AUC scores, F-1 scores or any other relevant metric.


**Data Source:** Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.

