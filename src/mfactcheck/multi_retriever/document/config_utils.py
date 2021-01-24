"""Parse configs for the Document Retriever module"""

from configs.config import Config


def _get_doc_configs(args):
    from argparse import Namespace

    _args = Namespace()

    setattr(_args, "dataset", "dev_fair")
    setattr(_args, "db_path", Config.db_path)
    setattr(_args, "document_k_wiki", Config.document_k_wiki)
    setattr(_args, "raw_training_set", Config.raw_training_set)
    setattr(_args, "document_add_claim", Config.document_add_claim)
    setattr(_args, "document_parallel", Config.document_parallel)
    setattr(_args, "train_ro_doc_file", Config.train_ro_doc_file)
    setattr(_args, "add_ro", Config.add_ro)
    setattr(_args, "document_ro_k_wiki", Config.document_ro_k_wiki)
    setattr(_args, "raw_dev_set", Config.raw_dev_set)
    setattr(_args, "dev_doc_file", Config.dev_doc_file)
    setattr(_args, "dev_ro_doc_file", Config.dev_ro_doc_file)
    setattr(_args, "raw_test_set", Config.raw_test_set)
    setattr(_args, "test_doc_file", Config.test_doc_file)
    setattr(_args, "training_doc_file", Config.training_doc_file)

    # update configs with passed args
    for k, v in (vars(args)).items():
        setattr(_args, k, v)
    return _args
