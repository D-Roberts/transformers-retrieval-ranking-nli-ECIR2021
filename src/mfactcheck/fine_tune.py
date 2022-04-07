"""
Multilingual evidence retrieval and fact verification codes.

Code references for the repo:
https://github.com/huggingface/transformers
https://github.com/UKPLab/fever-2018-team-athene
https://github.com/sheffieldnlp/fever-naacl-2018
https://github.com/ASoleimaniB/BERT_FEVER
"""

import argparse
import os

from mfactcheck.configs.config import Config
from mfactcheck.multi_nli.config_util import _get_nli_configs
from mfactcheck.multi_nli.predict import predict as nli_predict
from mfactcheck.multi_nli.train import train as nli_train
from mfactcheck.multi_retriever.document.config_utils import _get_doc_configs
from mfactcheck.multi_retriever.document.document_retrieval import (
    main as document_retrieval_main,
)
from mfactcheck.multi_retriever.document.ro_document_retrieval import (
    main as ro_document_retrieval_main,
)
from mfactcheck.multi_retriever.sentences.config_util import _get_sent_configs
from mfactcheck.multi_retriever.sentences.predict import (
    predict as sentence_selector_predict,
)
from mfactcheck.multi_retriever.sentences.train import train as sentence_selector_train
from mfactcheck.utils.log_helper import LogHelper


def document_retrieval(logger, args):
    args = _get_doc_configs(args)
    # setattr(_args, 'dataset', dataset)
    # setattr(_args, 'add_ro', add_ro)
    if args.dataset == "train" or args.task == "train":
        logger.info("Starting document retrieval for training set...")
        document_retrieval_main(
            args.db_path,
            args.document_k_wiki,
            args.raw_training_set,
            args.training_doc_file,
            args.document_add_claim,
            args.document_parallel,
        )
        logger.info("Finished document retrieval for training set.")
        if args.add_ro:
            logger.info("Starting document retrieval for Romanian train set...")
            ro_document_retrieval_main(
                args.document_ro_k_wiki, args.training_doc_file, args.train_ro_doc_file
            )
            logger.info("Finished document retrieval for Romanian train set.")

    elif args.task == "predict":  # predict
        if args.dataset in {"dev_fair", "dev_golden"}:
            logger.info("Starting document retrieval for English dev set...")
            document_retrieval_main(
                args.db_path,
                args.document_k_wiki,
                args.raw_dev_set,
                args.dev_doc_file,
                args.document_add_claim,
                args.document_parallel,
            )
            logger.info("Finished document retrieval for English dev set.")
            if args.add_ro:
                logger.info("Starting document retrieval for Romanian dev set...")
                ro_document_retrieval_main(
                    args.document_ro_k_wiki, args.dev_doc_file, args.dev_ro_doc_file
                )
                logger.info("Finished document retrieval for Romanian dev set.")
        else:  # get docs to predict Fever test set
            logger.info("Starting document retrieval for test set...")
            document_retrieval_main(
                args.db_path,
                args.document_k_wiki,
                args.raw_test_set,
                args.test_doc_file,
                args.document_add_claim,
                args.document_parallel,
            )
            logger.info("Finished document retrieval for test set.")


def sentence_retrieval(logger, args):
    args = _get_sent_configs(args)

    if args.task == "train":
        if not os.path.isdir(args.data_dir):
            os.makedirs(args.data_dir)
        logger.info("Starting sentence retrieval training...")
        sentence_selector_train(logger, args)
        # cleanup
        os.remove(args.train_tsv_file_pos)
        os.remove(args.train_tsv_file_neg)
        os.remove(os.path.join(args.data_dir, args.train_sentence_file_eval))
        logger.info("Finished sentence retrieval for training set.")
    else:  # predict
        logger.info("Starting sentence retrieval for dev/test set...")
        sentence_selector_predict(logger, args)
        os.remove(os.path.join(args.data_dir, args.predict_sentence_file_name))
        logger.info("Finished sentence retrieval for dev/test set.")


def nli(logger, trainer, args):
    args = _get_nli_configs(args)
    if args.task == "train":
        logger.info("Starting training fact validation module...")
        nli_train(logger, args)
        logger.info("Finished training fact validation module.")
    else:  # predict
        logger.info("Starting predicting fact validation module...")
        nli_predict(logger, trainer, args)
        logger.info("Finished predicting fact validation module.")


def main(args, logger):
    # debugging (sub-samples of datasets)
    logger.info(
        "=========================== Sub-task 1.0 Document Retrieval =========================================="
    )
    document_retrieval(logger, args)

    logger.info(
        "=========================== Sub-task 2.0 Sentence Selection =========================================="
    )
    sentence_retrieval(logger, args)

    logger.info(
        "=========================== Sub-task 3.0 Fact Validation ============================================"
    )
    nli(logger, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="/path/to/config/file, in JSON format",
        required=False,
        type=str,
    )
    parser.add_argument("--task", help="train or predict", default="predict", type=str)
    parser.add_argument(
        "--add_ro",
        help="if to retrieve Romanian documents",
        default=False,
        type=bool,
    )
    parser.add_argument(
        "--dataset",
        help="which dataset: train, dev_fair, dev_golden, test",
        default="dev_fair",
        type=str,
    )
    parser.add_argument(
        "--onnx",
        default=False,
        help="If to use the optimized converted onnx model for sentence selector\
                             in predict mode",
        type=bool,
    )
    parser.add_argument(
        "--api", default=False, help="if pipeline used in api", type=bool
    )
    args = parser.parse_args()

    LogHelper.setup()
    logger = LogHelper.get_logger(os.path.splitext(os.path.basename(__file__))[0])

    # If alternative configs are given in json file
    if args.config is not None:
        Config.load_config(args.config)

    main(args, logger)
