# DSCI_522_Group_412

# Proposal: Credit application predictor

Author(s): Clark Alistair, Dimri Aakanksha, Jiang Yue

Demo of a data analysis project for DSCI 522 (Data Science workflows); a course in the Master of Data Science program at the University of British Columbia.

## Motivation and Research question

Algorithms are used on an everyday basis to decide the outcome of credit applications. The purpose of our analysis is to answer the question: **"Given certain personal features, will the person’s credit card application be approved or not?"**

## Description of the Data

The Credit approval dataset is taken from the archives of the machine learning repository of University of California, Irvine (UCI). A quick review of the dataset, shows that all of the values in the dataset have been converted to meaningless symbols to protect the confidentiality of the data. However, to make the analysis intuitive and easy to understand, we gave the variables working names based on the type of data. **Note:** This is an important limitation of our analysis that doesn't affect prediction accuracy, but makes it difficult to interpret the EDA and feature importances of our model.

More information about the dataset can be found [here](http://archive.ics.uci.edu/ml/datasets/credit+approval).

## Analysis Approach

For our analysis we took the following steps:

- Complete exploratory analysis by creating several data visualizations to understand the underlying data and relationships between variables
- Perform data transformations like appropriate handling of missing values, standardising numerical features, encoding categorical features etc.
- Apply predictive classification models to answer the research question, including Random Forests, XGBoost, and LGBM.

## Results

To communicate the results effectively we plan to provide tables of the the following ouput:
- AUC scores
- Precision and Recall
- any other relevant metric

In addition we plan to show the following figure:

- Confusion Matrix plot

## Usage

To replicate the analysis, clone this GitHub repository, install the dependencies listed below, and run the following commands at the command line/terminal from the root directory of this project:

```python src/download_data.py --url='http://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data' --file_path='data/raw.csv'```

```python src/wrangle_df.py --input=train.csv --output=clean-train.csv```    
```python src/wrangle_df.py --input=test.csv --output=clean-test.csv```    

```Rscript src/visualizations.R --train=clean-train.csv --out_dir='img/'```

``` python src/analysis.py --input1=clean-train.csv --input2=clean-test.csv --output='results/' ```

## Dependencies
Python 3.7.3 and Python packages:

docopt==0.6.2     
requests==2.22.0     
pandas==0.24.2      
scikit-learn==0.22.1 .    


**Data Source:** Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.

