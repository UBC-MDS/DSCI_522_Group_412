# author: Alistair Clark
# date: 2020-01-17

"""This script downloads and writes a file to your laptop for a specified URL.
This script takes a URL and a local file path as command line arguments.

Usage: download_data.py --url=<url> --file_path=<file_path>

Options:
--url=<url>              URL of the file to download.
--file_path=<file_path>  Path (including filename) of the csv file to write.
"""

import pandas as pd
from docopt import docopt

opt = docopt(__doc__)

def main(url, file_path):
    df = pd.read_csv(url, header=None)
    df.to_csv(file_path)

if __name__ == "__main__":
    main(opt["--url"], opt["--file_path"])