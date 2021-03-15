## I. Article: [Multilingual Evidence Retrieval and Fact Verification to Combat Global Disinformation: The Power of Polyglotism](https://arxiv.org/pdf/2012.08919.pdf)
```
@article{roberts2020multilingual,
  title={Multilingual Evidence Retrieval and Fact Verification to Combat Global Disinformation: The Power of Polyglotism},
  author={Roberts, Denisa AO},
  journal={arXiv preprint arXiv:2012.08919},
  year={2020}
}
```


## II. API (MVP for research illustration purpose only)
The end to end system will be served via a live API during the ECIR 2021 conference. Furthermore, it can be accessed locally via Docker. 

![Multilingual evidence retrieval and fact verification system.](/assets/pacepa_eg.png)


### Steps to access API at http://0.0.0.0:8080/ :

Requires up to 4Gb RAM. 

Easiest. Install [Docker](https://docs.docker.com/get-docker/) then run:
```
docker run --rm -p 8080:8080 -m 4g droberts1/fact-verification
```
(tested on MacOS Big Sur and Ubuntu18.04 64-bitX86)

Build locally:
1. Get repo:
```
git clone https://github.com/D-Roberts/multilingual-nli-ECIR2021.git
cd multilingual-nli-ECIR2021
```

2. Build Docker (CPU):
```
docker build -t multi_api:latest -f dockers/docker-api-cpu/Dockerfile .
```

3. Run docker with mapped ports:
```
docker run --rm --ipc=host -p 8080:8080 multi_api:latest
```
4. In your browser go to http://0.0.0.0:8080/ , provide a claim with recognizable named entities, and the pipeline will run as depicted in the diagram above. Entities will be parsed, documents (Wikipedia pages) will be retrieved in English, Romanian and Portuguese, summaries tokenized into sentences and scored by the sentence selector, top 5 sentences will be provided to fact verifier and final prediction aggregated.


5. To score other files on CPU, one can run the same docker container. The dataset to score can be provided in a mapped data volume (see the GPU instructions for specifics) and predictions will be in a mapped out_dir_rte (refined_preds.jsonl). (need to replace paths to user's specifics)

```
docker run -it --rm --ipc=host -p 8080:8080 -v $LOCALPATH/data:/mfactcheck/data -v $LOCALPATH/out_dir_rte:/mfactcheck/out_dir_rte multi_api:latest bash

root@6acc74271d7b:/mfactcheck# python3 src/pipeline.py

# or

root@6acc74271d7b:/mfactcheck# python3 src/mfactcheck/multi_nli/predict.py --predict_rte_file=translated_data.tsv
```

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

#3. Optionally, download intermediary datasets output by each module: docs from [athene](https://github.com/UKPLab/fever-2018-team-athene), sent train/predict, nli train/predict files (en and ro):
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
