#!/bin/bash
mkdir data/translated_data
wget -O data/translated_data/translated_data.zip https://claimtraindata.s3.amazonaws.com/translated_data.zip
unzip data/translated_data/translated_data.zip -d data/translated_data

# use any of the file as input to nli predict script