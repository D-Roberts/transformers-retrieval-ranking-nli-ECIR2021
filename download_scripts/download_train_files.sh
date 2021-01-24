mkdir data/train_files

wget -O data/train_files/train_data.zip http://claimtraindata.s3.amazonaws.com/train_data.zip
unzip data/train_files/train_data.zip -d data/train_files

# use any of the files as training data tsv files by copy to data_dir dir and updating configs
# this data was used in training for paper results