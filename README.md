## I. Article.


[Multilingual Evidence Retrieval and Fact Verification to Combat Global Disinformation: The Power of Polyglotism](https://arxiv.org/pdf/2012.08919.pdf)



## II. System for Spanish - English (WIP). 
Disclaimer: For research illustration purposes only.


![Multilingual evidence retrieval and fact verification system.](/assets/pacepa_eg.png)


### Steps to access API locally, at http://0.0.0.0:8080/ :

Requires up to 3Gb RAM. 
Tested on MacOS Big Sur and Ubuntu18.04 64-bitX86. Pretrained models will be downloaded from S3.

Build locally:
1. Get repo:
```
git clone https://github.com/D-Roberts/multilingual-nli-ECIR2021.git
cd multilingual-nli-ECIR2021
```

2. Build and run Docker (CPU) for inference (ORT):
```
docker-compose up --build
```


## III. To get the Romanian-English translated dataset (and readme file) for article replication:
```
bash scripts/download-translated-data.sh
```

If you use this repository please consider citing the paper: 

```
@article{roberts2020multilingual,
  title={Multilingual Evidence Retrieval and Fact Verification to Combat Global Disinformation: The Power of Polyglotism},
  author={Roberts, Denisa AO},
  journal={arXiv preprint arXiv:2012.08919},
  year={2020}
}
```

## IV. Dir structure:
```
.
├── LICENSE
├── README.md
├── assets
│   ├── 379.png
│   ├── dir_struct.txt
│   ├── pacepa_eg.png
│   └── redis_data.tar.gz
├── docker-compose.yml
├── dockers
│   ├── docker-api-cpu
│   │   └── Dockerfile
│   └── docker-gpu
│       └── Dockerfile
├── redis-data
│   ├── appendonly.aof
│   └── dump.rdb
├── requirements
│   └── requirements.txt
├── requirements.txt
├── scripts
│   ├── build_db.py
│   ├── download-dev-files.sh
│   ├── download-doc-files-athens.sh
│   ├── download-fever-data.sh
│   ├── download-train-files.sh
│   ├── download-translated-data.sh
│   ├── download-wiki-pages.sh
│   ├── evaluation_script.py
│   └── run-mfactcheck.sh
├── server
│   ├── __init__.py
│   ├── app.py
│   ├── gunicorn_starter.sh
│   ├── mfactchecker.py
│   ├── preload.py
│   ├── static
│   │   ├── claims-final.png
│   │   ├── pacepa.png
│   │   └── style.css
│   └── templates
│       ├── _formhelpers.html
│       ├── claimform.html
│       ├── results.html
│       ├── thanks.html
│       └── unknown_id.html
├── src
│   └── mfactcheck
│       ├── __init__.py
│       ├── configs
│       │   └── config.py
│       ├── end_to_end_fine_tune.py
│       ├── models
│       │   ├── __init__.py
│       │   ├── mbert.py
│       │   ├── nli_mbert.py
│       │   └── sent_mbert.py
│       ├── multi_nli
│       │   ├── config_util.py
│       │   ├── data.py
│       │   ├── predict.py
│       │   └── train.py
│       ├── multi_retriever
│       │   ├── document
│       │   │   ├── config_utils.py
│       │   │   ├── document_retrieval.py
│       │   │   └── ro_document_retrieval.py
│       │   └── sentences
│       │       ├── __init__.py
│       │       ├── config_util.py
│       │       ├── data.py
│       │       ├── predict.py
│       │       └── train.py
│       ├── pipelines
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── multi_doc.py
│       │   ├── multi_nli.py
│       │   └── multi_sent.py
│       ├── trainer.py
│       └── utils
│           ├── __init__.py
│           ├── dataset
│           │   ├── data_processor.py
│           │   ├── data_utils.py
│           │   ├── fever_doc_db.py
│           │   └── reader.py
│           ├── log_helper.py
│           ├── model_utils.py
│           └── predict_utils.py
└── tests
    └── test_nli.py
    ```
    
## Contact:
denisa.roberts[at]denisaroberts.me

## License
Apache License Version 2.0
