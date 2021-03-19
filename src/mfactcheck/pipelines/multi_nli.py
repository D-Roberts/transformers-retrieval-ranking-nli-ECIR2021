"""Pipeline onnx client NLI cpu"""

import collections
import csv
import json
import os
import numpy as np

from mfactcheck.multi_nli.data import NLIProcessor, convert_examples_to_features
from mfactcheck.multi_nli.config_util import _get_nli_configs
from .base import Pipeline
from mfactcheck.utils.log_helper import LogHelper
from mfactcheck.utils.predict_utils import predictions_aggregator

LogHelper.setup()
logger = LogHelper.get_logger(os.path.splitext(os.path.basename(__file__))[0])


class MultiNLIPipeline(Pipeline):
    def __init__(self, module="nli", args=None, args_parser=_get_nli_configs):
        super().__init__(module, args, args_parser)
        self.processor = NLIProcessor()
        self.label_list = self.processor.get_labels()
        self.num_labels = len(self.label_list)

    def __call__(self):
        """Classify verification label given input claim-sentence pairs"""

        eval_examples = self.processor.get_dev_examples(self.args.data_dir, self.args.predict_rte_file)
        # eval_examples = eval_examples[0:20]  # debugging
        self.num_eg = len(eval_examples)

        eval_data = convert_examples_to_features(
            eval_examples, self.label_list, self.args.max_seq_length, self.tokenizer, "classification"
        )
       
        preds, labels, new_guids, guids_map = super().__call__(eval_data, self.num_eg)
        preds = np.argmax(preds, axis=1)  # 0 = Support; 1 = Refute; 2 = NEI

        # Implements the logic rules to get one verification prediction per claim from 5 separate predictions
        predictions_aggregator(
            logger, self.args, preds, labels, new_guids, guids_map, compute_acc=False
        )
    
    