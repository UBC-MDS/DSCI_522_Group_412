# Docker file for DSCI_522_GROUP_412
# 
# January 2020

# Use continuumio/anaconda3 as base image
FROM continuumio/anaconda3 

# Install R
RUN apt-get update && \
    apt-get install r-base r-base-dev -y

# Install Python machine learning tools
RUN conda install scikit-learn && \
    conda install pandas && \
    conda install numpy && \
    conda install py-xgboost

RUN pip install lightgbm

# Install R machine learning tools
RUN conda install -c r r-tidyverse && \
    conda install -c r r-tidyr && \
    conda install -c anaconda requests&& \   
    conda install -c conda-forge r-readr && \
    conda install r-readr 
    # conda install -c conda-forge r-ggally

# Install docopt Python package
RUN /opt/conda/bin/conda install -y -c anaconda docopt

FROM rocker/tidyverse

RUN Rscript -e "install.packages('GGally')"
RUN Rscript -e "install.packages('cowplot')"

# Put Anaconda Python in PATH
ENV PATH="/opt/conda/bin:${PATH}"

CMD ["/bin/bash"]