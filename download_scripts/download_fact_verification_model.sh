
mkdir nli_model
wget -O nli_model/enmbert_fine_tuned_rte.zip http://claimtraindata.s3.amazonaws.com/enmbert_fine_tuned_rte.zip
unzip nli_model/enmbert_fine_tuned_rte.zip -d nli_model

# copy model artifacts to out_dir_rte when ready to run api or make predictions
# cp nli_model/model/* out_dir_rte
# This is the fine-tuned EnmBERT