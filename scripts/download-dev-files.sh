#!/bin/bash
mkdir -p data/dev_files

wget -O data/dev_files/dev_fair_data.zip http://claimtraindata.s3.amazonaws.com/dev_fair_datasets.zip
unzip data/dev_files/dev_fair_data.zip -d data/dev_files

# use any of the files as dev data tsv files by copy to data_dir dir and updating configs
# this data was used in evaluation for paper results; predictions also included in pickle format