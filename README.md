
## Codebase for the article

Requires up to 3Gb RAM. 
Tested on MacOS Big Sur and Ubuntu18.04 64-bitX86. Fine-tuned models will be downloaded from S3. Expect initial build time approx 15min < t <= 30min.

Build locally:
1. Get repo:
```
git clone https://github.com/D-Roberts/multilingual-nli-ECIR2021.git
cd multilingual-nli-ECIR2021
```

2. Build and run Docker (CPU) for inference (onnx runtime):
```
docker-compose up --build
```


## Dir structure:
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
  

## License
Apache License Version 2.0

## Article 

[Multilingual Evidence Retrieval and Natural Language Inference](https://arxiv.org/pdf/2012.08919.pdf)

