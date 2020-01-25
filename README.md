# DSCI_522_Group_412

# Proposal: Credit application predictor

Author(s): Clark Alistair, Dimri Aakanksha, Jiang Yue

Demo of a data analysis project for DSCI 522 (Data Science workflows); a course in the Master of Data Science program at the University of British Columbia.

## Motivation and Research question

Algorithms are used on an everyday basis to decide the outcome of credit applications. The purpose of our analysis is to answer the question,
"Given information on individual's credit application, will the application be approved or rejected?" 

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
  - We plan to analyse the dataset with classifiers like Decision Trees and Logistic regression
  - We also plan to explore ensembling techniques like combining Random Forests and Classification algorithms to obtain higher accuracy
  - To further improve the accuracy of our model we plan to use techniques such as K-fold cross validation split, Grid search cross validation etc.
  
Thus far we have performed some exploratory data analysis, and the report for that can be found [here](https://github.com/UBC-MDS/DSCI_522_Group_412/blob/master/src/eda.ipynb). 

## Results

To communicate the results effectively we plan to provide tables of the the following ouput:
- AUC scores
- F-1 scores 
- Predictor importance
- any other relevant metric

In addition we plan to show the following figure:

Confusion Matrix plot

## Usage

To replicate the analysis, clone this GitHub repository, install the dependencies listed below, and run the following commands at the command line/terminal from the root directory of this project:

```python src/download_data.py --url='http://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data' --file_path='data/raw.csv'```

```python src/wrangle_df.py --input=train.csv --output=clean-train.csv```    
```python src/wrangle_df.py --input=test.csv --output=clean-test.csv```    

## Dependencies
Python 3.7.3 and Python packages:
docopt==0.6.2 
requests==2.22.0  
pandas==0.24.2   
pandas_profiling==2.4.0 .  


**Data Source:** Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.

