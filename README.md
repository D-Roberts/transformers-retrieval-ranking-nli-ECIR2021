## Article: Multilingual Evidence Retrieval and Fact Verification to Combat Global Disinformation: The Power of Polyglotism
https://arxiv.org/pdf/2012.08919.pdf
```
@article{roberts2020multilingual,
  title={Multilingual Evidence Retrieval and Fact Verification to Combat Global Disinformation: The Power of Polyglotism},
  author={Roberts, Denisa AO},
  journal={arXiv preprint arXiv:2012.08919},
  year={2020}
}
```

## API
The end to end system will be served via a live API during the ECIR 2021 conference. Furthermore, it can be build and accessed locally at http://0.0.0.0:8080/ when built and run via Docker. 

![Multilingual evidence retrieval and fact verification system.](/assets/pacepa_eg.png)


### Steps to run the API at http://0.0.0.0:8080/  :

1. Get the git repo. cd into it.
2. Make directories and download models:

```
mkdir data/data_dir
mkdir data/fever
mkdir out_dir_sent
mkdir out_dir_rte

# Download the trained optimized onnx sentence selection model that will be run via onnxruntime. Then copy converted_optim_quant_sent.onnx to dir out_dir_sent.
source download_scripts/download_sentence_selection_model.sh

# Download the trained rte/nli fact validation model. Copy the model artifacts to out_dir_rte folder.
source download_scripts/download_fact_verification_model.sh
```
3. Build Docker (CPU):
```
docker build -t multi_api:latest -f dockers/docker-api-cpu/Dockerfile .
```

4. Run docker with mapped data volumes and ports, as such (replace your own paths):
```
docker run -it --rm --ipc=host -p 8080:8080 -v /Users/denisaroberts/workspace/multilingual_nli_ECIR2021/data/fever:/mfactcheck/data -v /Users/denisaroberts/workspace/multilingual_nli_ECIR2021/out_dir_sent:/mfactcheck/out_dir_sent -v /Users/denisaroberts/workspace/multilingual_nli_ECIR2021/out_dir_rte:/mfactcheck/out_dir_rte multi_api:latest python3 app.py
```

## Code Base
Please see repository directory structure in assets/dir_struct.txt.


For inquiries: denisa.roberts[at]denisaroberts.me


