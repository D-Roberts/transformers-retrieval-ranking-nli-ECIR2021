#!/bin/bash
mkdir -p data/data_dir
wget -O data/translated_data.zip https://claimtraindata.s3.amazonaws.com/translated_data.zip
unzip data/translated_data.zip -d data
cp data/translated_data/roro0_dev.tsv data/data_dir

# use any of the files as input to nli predict script 