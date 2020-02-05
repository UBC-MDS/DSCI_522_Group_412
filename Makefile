# credit approval data pipe
# author: Alistair Clark
# date: 2020-01-30
# Usage: see project README for instructions on running this analysis

all: img/numerical.png results/accuracy_report.csv results/classification_report.csv results/model_compare.csv results/roc.png doc/Report_final.md 

# download data and split train and test sets
data/test.csv data/train.csv : src/download_data.py
	python src/download_data.py --url='http://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data' --out_dir=data/

# wrangle training data
data/clean-train.csv : src/wrangle_df.py data/train.csv
	python src/wrangle_df.py --input=data/train.csv --output=data/clean-train.csv

# wrangle testing data
data/clean-test.csv : src/wrangle_df.py data/test.csv
	python src/wrangle_df.py --input=data/test.csv --output=data/clean-test.csv

# exploratory data analysis - visualize numerical and categorical predictors
img/numerical.png : src/visualizations.R data/clean-train.csv
	Rscript src/visualizations.R --train=data/clean-train.csv --out_dir=img/

# Run model and produce results
results/accuracy_report.csv results/classification_report.csv results/model_compare.csv results/roc.png : src/analysis.py data/clean-train.csv data/clean-test.csv
	python src/analysis.py --input1=data/clean-train.csv --input2=data/clean-test.csv --output=results/

# render report
doc/Report_final.md : doc/Report_final.Rmd doc/credit_refs.bib
	Rscript -e "rmarkdown::render('doc/Report_final.Rmd', output_format = 'github_document')"

clean: 
	rm -rf data/*
	rm -rf img/*
	rm -rf results/*
	rm -rf doc/Report_final.md doc/Report_final.html