## I. Article: Multilingual Evidence Retrieval and Fact Verification to Combat Global Disinformation: The Power of Polyglotism
https://arxiv.org/pdf/2012.08919.pdf
```
@article{roberts2020multilingual,
  title={Multilingual Evidence Retrieval and Fact Verification to Combat Global Disinformation: The Power of Polyglotism},
  author={Roberts, Denisa AO},
  journal={arXiv preprint arXiv:2012.08919},
  year={2020}
}
```

## II. API
The end to end system will be served via a live API during the ECIR 2021 conference. Furthermore, it can be build and accessed locally at http://0.0.0.0:8080/ when built and run via Docker. 

![Multilingual evidence retrieval and fact verification system.](/assets/pacepa_eg.png)


### Steps to run the API at http://0.0.0.0:8080/  :

1. Get repo:
```
git clone git@github.com:D-Roberts/multilingual_nli_ECIR2021.git.
cd multilingual_nli_ECIR2021
```

2.  Make directories and download models:

```
mkdir data/data_dir
mkdir data/fever
mkdir out_dir_sent
mkdir out_dir_rte

# Download the trained optimized onnx sentence selection model (EnmBERT) that will be run via onnxruntime. Then copy converted_optim_quant_sent.onnx to dir out_dir_sent.
source download_scripts/download_sentence_selection_model.sh

cp onnx_model/converted_optim_quant_sent.onnx out_dir_sent

# Download the trained rte/nli fact validation model (EnmBERT). Copy the model artifacts to out_dir_rte folder.
source download_scripts/download_fact_verification_model.sh

cp nli_model/model/* out_dir_rte
```
3. Build Docker (CPU):
```
docker build -t multi_api:latest -f dockers/docker-api-cpu/Dockerfile .
```

4. Run docker with mapped data volumes and ports, as such (replace your own paths):
```
docker run -it --rm --ipc=host -p 8080:8080 -v /Users/denisaroberts/workspace/multilingual_nli_ECIR2021/data/fever:/mfactcheck/data -v /Users/denisaroberts/workspace/multilingual_nli_ECIR2021/out_dir_sent:/mfactcheck/out_dir_sent -v /Users/denisaroberts/workspace/multilingual_nli_ECIR2021/out_dir_rte:/mfactcheck/out_dir_rte multi_api:latest python3 app.py
```
5. In your browser go to http://0.0.0.0:8080/ , provide a claim with recognizable named entities, and the pipeline will run as depicted in the diagram above. Entities will be parsed, documents (Wikipedia pages) will be retrieved in English, Romanian and Portuguese, summaries tokenized into sentences and scored by the sentence selector, top 5 sentences will be provided to fact verifier and final prediction aggregated.

```
docker run -it --rm --ipc=host -p 8080:8080 -v /Users/denisaroberts/workspace/multilingual_nli_ECIR2021/data/fever:/mfactcheck/data -v /Users/denisaroberts/workspace/multilingual_nli_ECIR2021/out_dir_sent:/mfactcheck/out_dir_sent -v /Users/denisaroberts/workspace/multilingual_nli_ECIR2021/out_dir_rte:/mfactcheck/out_dir_rte multi_api:latest bash

root@6acc74271d7b:/mfactcheck# python3 src/pipeline.py
# or 
root@6acc74271d7b:/mfactcheck# python3 src/mfactcheck/multi_nli/predict.py --predict_rte_file=translated_data.tsv
```
6. To score other files on CPU, one can run the same docker container. The dataset to score can be provided in that mapped data/data_dir and predictions will be in the mapped out_dir_rte (refined_preds.jsonl).

## III To get the Romanian-English translated dataset (and readme file):
```
source download_scripts/download_translated_data.sh
```

## IV. Code Base
Please see repository directory structure in [assets](https://github.com/D-Roberts/multilingual_nli_ECIR2021/blob/main/assets/dir_struct.txt).


### Steps to get train and dev files and train or predict on (1) GPU 

1. Build docker. At the repo root:
```
docker build -t mtest:latest -f dockers/docker-gpu/Dockerfile .
```
2. Get data and build datasets. Data is under Wikipedia and fever.ai associated licenses included in the downloads.
```
#1. Make directories and get FEVER task data (En)
source download_scripts/download_fever_data.sh

#2. Wikipedia pages used in the FEVER task (En)
source download_scripts/download_wiki_pages.sh

# Also build the db from within the docker env:
docker run -it --rm --ipc=host  -v /home/ubuntu/mfactcheck/data:/mfactcheck/data -v /home/ubuntu/mfactcheck/onnx_model:/mfactcheck/onnx_model -v /home/ubuntu/mfactcheck/out_dir_rte/:/mfactcheck/out_dir_rte -v /home/ubuntu/mfactcheck/out_dir_sent:/mfactcheck/out_dir_sent  mtest:latest

root@6acc74271d7b:/mfactcheck# python3 build_datasets_scripts/build_db.py data/wiki-pages data/fever/fever.db
```

#3. Optionally, download a variety of intermediary doc, sent train/predict, nli train/predict files (en and ro):
```
source download_scripts/download_doc_files_athens.sh
source download_scripts/download_train_files.sh
source download_scripts/download_dev_files.sh
```

## Contact:
denisa.roberts[at]denisaroberts.me

## License
Apache License Version 2.0
