## I. Article.
If you use this repository please consider citing the paper: 

[Multilingual Evidence Retrieval and Fact Verification to Combat Global Disinformation: The Power of Polyglotism](https://arxiv.org/pdf/2012.08919.pdf)
```
@article{roberts2020multilingual,
  title={Multilingual Evidence Retrieval and Fact Verification to Combat Global Disinformation: The Power of Polyglotism},
  author={Roberts, Denisa AO},
  journal={arXiv preprint arXiv:2012.08919},
  year={2020}
}
```


## II. API (MVP for research illustration purpose only). Live at: http://34.207.240.102:8080/

* Try fun claims such as {"COVID-19 was sent over by aliens from Mars."} Good way to retrieve facts and learn some Portuguese and Romanian!

* System runs at approx 20 QPS. Last step on cached pre-retrieved top 5 evidence sentences runs within 0.25s SLA; end-to-end approx 30s (all steps: constituency parsing, term searches with MediaWiki API in 3 languages (~12-20 terms); retrieve Wiki page summaries; ONNX runtime sentence scoring (100-500 sentences); top-5 natural language inference ONNX runtime scoring).

* If busy - reload. Scaled to limited capacity.

* The end to end system will be served via a live API during the ECIR 2021 conference. 

* Furthermore, it can be accessed locally via Docker.



![Multilingual evidence retrieval and fact verification system.](/assets/pacepa_eg.png)


### Steps to access API locally, at http://0.0.0.0:8080/ :

Requires up to 4Gb RAM. 
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


## III. To get the Romanian-English translated dataset (and readme file):
```
bash scripts/download-translated-data.sh
```


## Contact:
denisa.roberts[at]denisaroberts.me

## License
Apache License Version 2.0
