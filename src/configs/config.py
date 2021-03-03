import json
import os
import os.path as path


class Config:
    @classmethod
    def load_config(cls, conf_path):
        with open(conf_path) as f:
            conf = json.load(f)
            for k, v in conf.items():
                setattr(cls, k, v)

    @classmethod
    def save_config(cls, conf_path):
        obj = {}
        for k, v in cls.__dict__.items():
            if not isinstance(v, classmethod) and not k.startswith("__"):
                obj.update({k: v})
        with open(conf_path, "w") as f:
            json.dump(obj, f, indent=4)

    @classmethod
    def get_args(cls):
        obj = {}
        for k, v in cls.__dict__.items():
            if not isinstance(v, classmethod) and not isinstance(v, dict) and not k.startswith("__"):
                obj.update({k: v})
        return obj

    BASE_DIR = os.getcwd()

    # Original FEVER task
    data_top = path.join(BASE_DIR, "data")
    db_path = path.join(BASE_DIR, "data/fever/fever.db")
    dataset_folder = path.join(BASE_DIR, "data/fever")

    raw_training_set = path.join(BASE_DIR, "data/fever-data/train.jsonl")
    raw_dev_set = path.join(BASE_DIR, "data/fever-data/dev.jsonl")
    raw_test_set = path.join(BASE_DIR, "data/fever-data/test.jsonl")

    # Doc retrieval
    training_doc_file = path.join(dataset_folder, "train.wiki7.jsonl")
    dev_doc_file = path.join(dataset_folder, "dev.wiki7.jsonl")
    test_doc_file = path.join(dataset_folder, "test.wiki7.jsonl")

    dev_ro_doc_file = path.join(dataset_folder, "ro_dev.wiki1.jsonl")
    train_ro_doc_file = path.join(dataset_folder, "ro_train.wiki1.jsonl")


    # Sentence Selector and Fact Checker data dir and files
    data_dir = path.join(BASE_DIR, "data/data_dir")
    # cached docs for api run
    cached_docs = path.join(data_dir, "en_ro_pt_docs.jsonl")
    

    # For model runs
    cache_dir = ""

    train_tsv_file_pos = path.join(data_dir, "train_sent_pos.tsv")
    train_tsv_file_neg = path.join(data_dir, "train_sent_neg_32.tsv")

    train_rte_file = path.join(data_dir, "train_rte_file.tsv")
    predict_rte_file = path.join(data_dir, "predict_rte_file.tsv")

    # If EnmBERT or EnRomBERT pipeline
    add_ro = False

    # If using onnx converted models for prediction
    onnx = True

    # If running the API
    api = False

    # Doc retrieval
    document_k_wiki = 7
    document_ro_k_wiki = 1
    document_parallel = True
    document_add_claim = True

    # Model dirs
    out_dir_sent = path.join(BASE_DIR, "out_dir_sent")
    out_dir_rte = path.join(BASE_DIR, "out_dir_rte")

    # Sent proc
    sent_param = {
        "sent_k": 5,
        "do_doc_process": True,
        "num_neg_samples": 2,
        "num_ro_samples": 2,
    }
    # eval dataset options: 'dev_fair', 'dev_gold', 'test', 'train'
    dataset = 'test'
    task = 'predict'

    # Models configs
    model_params_sentence = {
        "bert_model": "bert-base-multilingual-cased",
        "max_seq_length": 128,
        "do_lower_case": False,
        "train_batch_size": 4, # 32
        "negative_batch_size": 4, # 32
        "losstype": "cross_entropy",
        "eval_batch_size": 1,
        "learning_rate": 2e-5,
        "num_train_epochs": 1,
        "warmup_proportion": 0.1,
        "gradient_accumulation_steps": 1,
    }

    model_params_nli = {
        "bert_model": "bert-base-multilingual-cased",
        "max_seq_length": 128,
        "do_lower_case": False,
        "train_batch_size": 4, # 32
        "losstype": "cross_entropy",
        "eval_batch_size": 1,
        "learning_rate": 2e-5,
        "num_train_epochs": 2,
        "warmup_proportion": 0.1,
        "gradient_accumulation_steps": 1,
    }

    # Trainer configs
    trainer_params = {"seed": 42, "local_rank": -1, "no_cuda": False}
