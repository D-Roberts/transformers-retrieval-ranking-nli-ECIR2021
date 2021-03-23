"""NLI predict CPU or 1 GPU"""
import argparse
import collections
import csv
import json
import os

import numpy as np

from mfactcheck.multi_nli.config_util import _get_nli_configs
from mfactcheck.models.nli_mbert import NLIMBert
from mfactcheck.multi_nli.data import convert_examples_to_features
from mfactcheck.trainer import Trainer
from mfactcheck.utils.log_helper import LogHelper
from mfactcheck.utils.predict_utils import predictions_aggregator


def predict(logger, args):
    """Predict script for NLI module"""

    module = NLIMBert(args.output_dir)
    model = module.model
    label_list = module.label_list
    num_labels = module.num_labels
    eval_examples = module.processor.get_dev_examples(
        args.data_dir, args.predict_rte_file
    )
    # eval_examples = eval_examples[0:20]  # debugging
    num_eg = len(eval_examples)

    eval_data = convert_examples_to_features(
        eval_examples,
        label_list,
        args.max_seq_length,
        module.tokenizer,
        "classification",
    )

    args.onnx = False  # used in pipelines
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
