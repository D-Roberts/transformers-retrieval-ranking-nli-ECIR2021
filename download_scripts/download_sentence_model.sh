
mkdir sentence_model
wget -O sentence_model/enmbert_sentence.zip http://claimtraindata.s3.amazonaws.com/sentence_selection_model.zip
unzip sentence_model/enmbert_sentence.zip -d sentence_model

# copy model artifacts to out_dir_sent when ready to run api or make predictions
# cp sentence_model/sentence_selection_model/* out_dir_sent
# This is the fine-tuned EnmBERT for sentence selection