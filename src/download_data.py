# author: Alistair Clark
# date: 2020-01-17

"""This script downloads and writes a file to your laptop for a specified URL.
This script takes a URL and a directory path as command line arguments.

Usage: download_data.py --url=<url> --out_dir=<out_dir>

Options:
--url=<url>         URL from where to download the data (must be in standard csv format)
--out_dir=<out_dir> Path to directory where the processed data will be written.
"""

import pandas as pd
from docopt import docopt
from sklearn.model_selection import train_test_split
import requests
from requests.exceptions import HTTPError
import sys

opt = docopt(__doc__)

def main(url, out_dir):
    # Check url exists
    try:
        request = requests.get(url)
        request.raise_for_status()
    except HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        sys.exit()
    except Exception as err:
        print(f"Other error occurred: {err}")
        sys.exit()

    # Read csv and split into train and test
    df = pd.read_csv(url, header=None)
    train_df, test_df = train_test_split(df, test_size = 0.2, random_state=42)

    # Write train and test .csv files
    try:
        train_df.to_csv(out_dir + "train.csv")
        test_df.to_csv(out_dir + "test.csv")
    except Exception as e:
        print("Directory does not exist.")
    
if __name__ == "__main__":
    main(opt["--url"], opt["--out_dir"])