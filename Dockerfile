# Docker file for DSCI_522_GROUP_412
# 
# January 2020

# Use continuumio/anaconda3 as base image
FROM continuumio/anaconda3 

# Install base R
RUN apt-get update && \
    apt-get install r-base r-base-dev -y

# Install Python packages
RUN conda install -c conda-forge xgboost==0.90 && \
    conda install -c conda-forge lightgbm==2.3.0 && \
    conda install -c anaconda requests && \   
    conda install -y -c anaconda docopt && \
    conda install -c conda-forge ipython-autotime

# Upgrade scikit-learn to 0.22.1
RUN pip install -U scikit-learn

# Install R packages
RUN conda install -y -c r r-tidyverse==1.2.1 && \
    conda install -y -c conda-forge r-docopt==0.6.1 && \
    conda install -c conda-forge r-ggally  && \
    conda install -c conda-forge r-cowplot

# Put Anaconda Python in PATH
ENV PATH="/opt/conda/bin:${PATH}"

CMD ["/bin/bash"]