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
    conda install numpy 

RUN conda install --quiet --yes \
    'boost' \
    'lightgbm' \
    'xgboost'  && \
    conda clean -tipsy && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

# Install R machine learning tools
RUN conda install -c r r-tidyverse && \
    conda install -c r r-tidyr && \
    conda install -c anaconda requests&& \   
    conda install -c conda-forge r-readr && \
    conda install r-readr

# Install docopt Python package
RUN /opt/conda/bin/conda install -y -c anaconda docopt

# Put Anaconda Python in PATH
ENV PATH="/opt/conda/bin:${PATH}"