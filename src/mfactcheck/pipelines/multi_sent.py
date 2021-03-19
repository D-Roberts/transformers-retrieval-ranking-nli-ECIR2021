"""Pipeline onnx client Sentence Selector cpu"""

import collections
import csv
import json
import os
import numpy as np

from mfactcheck.multi_retriever.sentences.data import (
    SentenceProcessor,
    get_eval_data,
    get_topk_sentences_eval,
)
from mfactcheck.multi_retriever.sentences.config_util import _get_sent_configs
from .base import Pipeline
from mfactcheck.utils.log_helper import LogHelper

LogHelper.setup()
logger = LogHelper.get_logger(os.path.splitext(os.path.basename(__file__))[0])


class MultiSentPipeline(Pipeline):
    def __init__(self, module="sent", args=None, args_parser=_get_sent_configs):
        super().__init__(module, args, args_parser)
        self.processor = SentenceProcessor()
        self.label_list = self.processor.get_labels()
        self.num_labels = len(self.label_list)
        self.label_verification_list = self.processor.get_labels_verification()

    def __call__(self):
        """Classify verification label given input claim-sentence pairs"""

        eval_features, self.num_eg = get_eval_data(
            logger,
            db_path=self.args.db_path,
            data_dir=self.args.data_dir,
            output_file_name=self.args.predict_sentence_file_name,
            processor=self.processor,
            tokenizer=self.tokenizer,
            output_mode="classification",
            label_list=self.label_list,
            label_verification_list=self.label_verification_list,
            max_seq_length=self.args.max_seq_length,
            dataset=self.args.dataset,
            do_doc_process=self.args.do_doc_process,
            add_ro=self.args.add_ro,
            doc_file=self.args.test_doc_file,
            ro_doc_file=self.args.dev_ro_doc_file,
            api=self.args.api,
        )
       
        logits, _, new_guids, guids_map = super().__call__(eval_features, self.num_eg)
         # topk selector: get dataset for nli module (dev, test)
        get_topk_sentences_eval(
            zip(new_guids, logits),
            guids_map,
            os.path.join(self.args.data_dir, self.args.predict_sentence_file_name),
            self.args.predict_rte_file,
            self.args.sent_k,
        )
