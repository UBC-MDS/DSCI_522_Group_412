# credit approval data pipe
# author: Alistair Clark
# date: 2020-01-30
# Usage: see project README for instructions on running this analysis

all: data/test.csv data/train.csv data/clean-test.csv data/clean-train.csv img/categorical.png img/numerical.png results/accuracy_report.csv results/classification_report.csv results/model_compare.csv results/roc.png doc/Report_final.md 

# download data and split train and test sets
data/test.csv data/train.csv : src/download_data.py
	python src/download_data.py --url='http://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data' --out_dir=data/

# wrangle training data
data/clean-train.csv : src/wrangle_df.py
	python src/wrangle_df.py --input=train.csv --output=clean-train.csv

# wrangle testing data
data/clean-test.csv : src/wrangle_df.py
	python src/wrangle_df.py --input=test.csv --output=clean-test.csv

# exploratory data analysis - visualize numerical and categorical predictors
img/categorical.png img/numerical.png : src/visualizations.R
	Rscript src/visualizations.R --train=clean-train.csv --out_dir=img/

# Run model and produce results
results/accuracy_report.csv results/classification_report.csv results/model_compare.csv results/roc.png : src/analysis.py
	python src/analysis.py --input1=clean-train.csv --input2=clean-test.csv --output=results/

# render report
doc/Report_final.md : doc/Report_final.Rmd doc/credit_refs.bib
	Rscript -e "rmarkdown::render('doc/Report_final.Rmd', output_format = 'github_document')"

clean: 
	rm -rf data/*
	rm -rf img/*
	rm -rf results/*
	rm -rf doc/Report_final.md doc/Report_final.html