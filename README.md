## I. Article: [Multilingual Evidence Retrieval and Fact Verification to Combat Global Disinformation: The Power of Polyglotism](https://arxiv.org/pdf/2012.08919.pdf)
```
@article{roberts2020multilingual,
  title={Multilingual Evidence Retrieval and Fact Verification to Combat Global Disinformation: The Power of Polyglotism},
  author={Roberts, Denisa AO},
  journal={arXiv preprint arXiv:2012.08919},
  year={2020}
}
```


## II. API (research illustration purpose only)
The end to end system will be served via a live API during the ECIR 2021 conference. Furthermore, it can be accessed locally when built and run via Docker. 

![Multilingual evidence retrieval and fact verification system.](/assets/pacepa_eg.png)


### Steps to access API at http://0.0.0.0:8080/  :

1. Get repo:
```
git clone https://github.com/D-Roberts/multilingual-nli-ECIR2021.git
cd multilingual-nli-ECIR2021
```

2.  Make directories and download models:

```
mkdir -p data/data_dir
mkdir -p data/fever
mkdir out_dir_sent
mkdir out_dir_rte

# Download the trained optimized onnx sentence selection model (EnmBERT) that will be run via onnxruntime. Then copy converted_optim_quant_sent.onnx to dir out_dir_sent.
bash scripts/download-sentence-model.sh
cp sentence_model/sentence_selection_model/* out_dir_sent

# Download the trained rte/nli fact validation model (EnmBERT). Copy the model artifacts to out_dir_rte folder.
bash scripts/download-fact-verification-model.sh
cp nli_model/model/* out_dir_rte
```
3. Build Docker (CPU):
```
docker build -t multi_api:latest -f dockers/docker-api-cpu/Dockerfile .
```

4. Run docker with mapped data volumes and ports, as such (replace your own paths):
```
docker run -it --rm --ipc=host -p 8080:8080 -v /Users/denisaroberts/workspace/multilingual-nli-ECIR2021/data:/mfactcheck/data -v /Users/denisaroberts/workspace/multilingual-nli-ECIR2021/out_dir_sent:/mfactcheck/out_dir_sent -v /Users/denisaroberts/workspace/multilingual-nli-ECIR2021/out_dir_rte:/mfactcheck/out_dir_rte multi_api:latest python3 app.py
```
5. In your browser go to http://0.0.0.0:8080/ , provide a claim with recognizable named entities, and the pipeline will run as depicted in the diagram above. Entities will be parsed, documents (Wikipedia pages) will be retrieved in English, Romanian and Portuguese, summaries tokenized into sentences and scored by the sentence selector, top 5 sentences will be provided to fact verifier and final prediction aggregated.

```
docker run -it --rm --ipc=host -p 8080:8080 -v /Users/denisaroberts/workspace/multilingual-nli-ECIR2021/data:/mfactcheck/data -v /Users/denisaroberts/workspace/multilingual-nli-ECIR2021/out_dir_sent:/mfactcheck/out_dir_sent -v /Users/denisaroberts/workspace/multilingual-nli-ECIR2021/out_dir_rte:/mfactcheck/out_dir_rte multi_api:latest bash

root@6acc74271d7b:/mfactcheck# python3 src/pipeline.py
# or 
root@6acc74271d7b:/mfactcheck# python3 src/mfactcheck/multi_nli/predict.py --predict_rte_file=translated_data.tsv
```
6. To score other files on CPU, one can run the same docker container. The dataset to score can be provided in that mapped data/data_dir and predictions will be in the mapped out_dir_rte (refined_preds.jsonl).


## III. To get the Romanian-English translated dataset (and readme file):
```
bash scripts/download-translated-data.sh
```


## IV. Codebase
Please see repository directory structure in [assets](https://github.com/D-Roberts/multilingual-nli-ECIR2021/blob/main/assets/dir_struct.txt).


### Steps to get train and dev files and train or predict on (1) GPU 

1. Build docker. At the repo root:
```
docker build -t mtest:latest -f dockers/docker-gpu/Dockerfile .
```
2. Get data and build datasets. Data is under Wikipedia and [fever.ai](https://fever.ai/) associated licenses included in the downloads.
```
#1. Make directories and get FEVER task data (En)
bash scripts/download-fever-data.sh

#2. Wikipedia pages used in the FEVER task (En)
bash scripts/download-wiki-pages.sh

# Also build the db from within the docker env:
docker run -it --rm --ipc=host -v /home/ubuntu/multilingual-nli-ECIR2021/data:/mfactcheck/data -v /home/ubuntu/multilingual-nli-ECIR2021/out_dir_rte/:/mfactcheck/out_dir_rte -v /home/ubuntu/multilingual-nli-ECIR2021/out_dir_sent:/mfactcheck/out_dir_sent  mtest:latest

root@6acc74271d7b:/mfactcheck# python3 scripts/build_db.py data/wiki-pages data/fever/fever.db
```

#3. Optionally, download a variety of intermediary docs, docs from [athene](https://github.com/UKPLab/fever-2018-team-athene), sent train/predict, nli train/predict files (en and ro):
```
bash scripts/download-doc-files-athens.sh
bash scripts/download-train-files.sh
bash scripts/download-dev-files.sh
```
#4. Train / predict the full pipeline or intermediary components. Datasets must be in data/data_dir mapped to docker volume. In training, model dirs out_dir_sent and out_dir_rte are recreated.
```
root@6acc74271d7b:/mfactcheck# python3 src/pipeline.py --task=train

# or
root@6acc74271d7b:/mfactcheck# python3 src/mfactcheck/multi_nli/train.py --[options]
```


## Contact:
denisa.roberts[at]denisaroberts.me

## License
Apache License Version 2.0
