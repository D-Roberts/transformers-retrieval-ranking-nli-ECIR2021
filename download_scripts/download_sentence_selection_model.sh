mkdir onnx_model
wget -O onnx_model/sentence_selection_model_onnx.zip http://claimtraindata.s3.amazonaws.com/sentence_selection_model_onnx.zip
unzip onnx_model/sentence_selection_model_onnx.zip -d onnx_model

# copy model to out_dir_sent when ready to run api or make predictions