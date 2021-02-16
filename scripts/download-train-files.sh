#!/bin/bash
wget -O data/train_files.zip https://claimtraindata.s3.amazonaws.com/train_datasets.zip
unzip data/train_files.zip -d data/train_files
rm data/train_files.zip

# use any of the files as training data tsv files by copy to data_dir dir and updating configs
# this data was used in training for paper results