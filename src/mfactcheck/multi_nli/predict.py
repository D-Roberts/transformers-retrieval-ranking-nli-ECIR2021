"""NLI predict CPU or 1 GPU"""
import argparse
import collections
import csv
import json
import os

import numpy as np
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.tokenization import BertTokenizer

from mfactcheck.multi_nli.config_util import _get_nli_configs
from mfactcheck.multi_nli.data import NLIProcessor, convert_examples_to_features
from mfactcheck.trainer import Trainer
from mfactcheck.utils.log_helper import LogHelper
from mfactcheck.utils.model_utils import get_model_dir
from mfactcheck.utils.dataset.data_utils import _clean_last as clean
from mfactcheck.utils.predict_utils import predictions_aggregator


def predict(logger, args):

    processor = NLIProcessor()
    output_mode = "classification"
    args.onnx = False

    # load/download the right model
    if not os.path.isdir(args.output_dir):
        get_model_dir(args.output_dir, args.add_ro, "nli", args.onnx)

    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load the trained model
    model = None
    model = BertForSequenceClassification.from_pretrained(
        args.output_dir, num_labels=num_labels
    )
    tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=False)

    eval_examples = processor.get_dev_examples(args.data_dir, args.predict_rte_file)
    # eval_examples = eval_examples[0:20]  # debugging
    num_eg = len(eval_examples)

    eval_data = convert_examples_to_features(
        eval_examples, label_list, args.max_seq_length, tokenizer, output_mode
    )

    trainer = Trainer(model=model, args=args)
    preds, labels, new_guids, guids_map = trainer.predict(eval_data, num_eg)
    preds = np.argmax(preds, axis=1)  # 0 = Support; 1 = Refute; 2 = NEI

    # Implements the logic rules to get one verification prediction per claim from 5 separate predictions
    predictions_aggregator(
        logger, args, preds, labels, new_guids, guids_map, compute_acc=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predict-rte-file",
        type=str,
        required=True,
        help="Input file in tsv format loaded from data_dir",
    )
    parser.add_argument(
        "--translated",
        type=bool,
        default=True,
        help="if a separate input file is provided",
    )
    parser.add_argument("--add_ro", type=bool, default=False)
    args = parser.parse_args()
    args = _get_nli_configs(args)
    LogHelper.setup()
    # can score separately
    logger = LogHelper.get_logger(os.path.splitext(os.path.basename(__file__))[0])
    predict(logger, args)
