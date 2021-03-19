import csv
import json
import os

from mfactcheck.multi_retriever.document.api_doc_retrieval import main as doc_retrieval
from mfactcheck.multi_retriever.sentences.config_util import _get_sent_configs
from mfactcheck.pipeline import sentence_retrieval
from mfactcheck.configs.config import Config


def run_document_retrieval():
    
    doc_path = Config.test_doc_file

    doc_retrieval(
        3,
        os.path.join(Config.data_dir, "input.jsonl"),
        os.path.join(doc_path),
    )


def run_evidence_recommendation(logger):
    from argparse import Namespace

    args = Namespace()
    args = _get_sent_configs(args)
    sentence_retrieval(logger, args)
