#!/bin/bash
mkdir -p data/dev_files

wget -O data/dev_files.zip https://claimtraindata.s3.amazonaws.com/fair_dev_datasets.zip
unzip data/dev_files.zip -d data/dev_files
rm -r data/dev_files.zip
# use any of the files as dev data tsv files by copy to data_dir dir and updating configs
# this data was used in evaluation for paper results; predictions also included in pickle format