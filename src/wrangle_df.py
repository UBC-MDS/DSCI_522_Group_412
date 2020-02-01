# author: Yue Jiang
# date: 2020-01-21

'''This script wrangles UCI Credit Approval data set and saves to a new file.

Usage: src/wrangle_df.py --input=<input> --output=<output>

Options:
--input=<input>  Name of csv file to wrangle, must be within the /data directory.
--output=<output>  Name of clean dataset file to be saved in /data directory. 
'''

import pandas as pd
import numpy as np
from docopt import docopt

opt = docopt(__doc__)


def main(input, output):
    # download xls to pandas dataframe
    df = pd.read_csv(f"./data/{input}")

    # test that the data is a pandas data frame object
    assert isinstance(df, pd.core.frame.DataFrame)
    
    #test that the data frame has 16 features columns and 1 index column
    assert len(df.columns) == 17

    # drop unecessary index column
    df = df.drop(df.columns[0], 1)

    # Add column names
    new_col_names = ['Sex',
                    'Age',
                    'Debt',
                    'Married',
                    'BankCustomer',
                    'EducationLevel',
                    'Ethnicity',
                    'YearsEmployed',
                    'PriorDefault',
                    'Employed',
                    'CreditScore',
                    'DriversLicense',
                    'Citizen',
                    'ZipCode',
                    'Income',
                    'Approved'
                    ]

    df.columns = new_col_names

    # Dealing with missing data identified as "?"
    df = df.replace('?', np.nan)

    # Change age from string to float
    df.Age = df.Age.astype(float)

    # Dealing with missing data
    # Numerical Variable Age - Substituting with median values
    df.fillna(df.median(), inplace=True)
   
    # For Categorical features - replace missing values with the most common occuring category value
    df = df.apply(lambda x: x.fillna(x.value_counts().index[0]))

    # Change PriorDefault to False and True
    df['PriorDefault'] = df['PriorDefault'].replace({'f': 'False', 't': 'True'})

    # Change Employed to False and True
    df['Employed'] = df['Employed'].replace({'f': 'False', 't': 'True'})

    # Change target to 0 and 1
    df['Approved'] = df['Approved'].replace({'-': 0, '+': 1})

    # Make sure the target only contains values 0 or 1
    assert df['Approved'].isin([0, 1]).all()

    # Save cleaned data to file
    df.to_csv(r"./data/%s" % (output), index=False)


if __name__ == "__main__":
    main(input=opt["--input"], output=opt["--output"])