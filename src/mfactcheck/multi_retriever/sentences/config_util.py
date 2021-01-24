"""Parse configs for the Sentence Selector module"""

from configs.config import Config


def _get_sent_configs(args):
    from argparse import Namespace

    _args = Namespace()
    for k, v in Config.model_params_sentence.items():
        setattr(_args, k, v)
    for k, v in Config.trainer_params.items():
        setattr(_args, k, v)
    for k, v in Config.sent_param.items():
        setattr(_args, k, v)

    setattr(_args, "onnx", False)
    setattr(_args, "api", False)
    setattr(_args, "dataset", "dev_fair")
    setattr(_args, "cache_dir", "")
    setattr(_args, "data_dir", Config.data_dir)
    setattr(_args, "db_path", Config.db_path)
    setattr(_args, "output_dir", Config.out_dir_sent)

    setattr(_args, "dev_doc_file", Config.dev_doc_file)
    setattr(_args, "test_doc_file", Config.test_doc_file)
    setattr(_args, "train_doc_file", Config.training_doc_file)

    setattr(_args, "dev_ro_doc_file", Config.dev_ro_doc_file)
    setattr(_args, "add_ro", Config.add_ro)
    setattr(_args, "train_ro_doc_file", Config.train_ro_doc_file)

    setattr(_args, "train_sentence_file_eval", "train_sent_file_eval.tsv")
    setattr(_args, "pos_sent_train", Config.train_tsv_file_pos)
    setattr(_args, "neg_sent_train", Config.train_tsv_file_neg)
    setattr(_args, "predict_sentence_file_name", "predict_sent_file.tsv")

    setattr(_args, "train_rte_file", Config.train_rte_file)
    setattr(_args, "predict_rte_file", Config.predict_rte_file)

    # update configs with passed args
    for k, v in (vars(args)).items():
        setattr(_args, k, v)
    return _args
