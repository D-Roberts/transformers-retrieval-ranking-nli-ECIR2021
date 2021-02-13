"""Sentence Selector Predict/Eval CPU or 1 GPU"""

import argparse
import os

from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.tokenization import BertTokenizer

from mfactcheck.multi_retriever.sentences.config_util import _get_sent_configs
from mfactcheck.multi_retriever.sentences.data import (
    SentenceProcessor,
    get_eval_data,
    get_topk_sentences_eval,
)
from trainer import Trainer
from utils.log_helper import LogHelper
from utils.file_utils import get_model_dir


def predict(logger, args):

    processor = SentenceProcessor()
    output_mode = "classification"


    label_list = processor.get_labels()
    label_verification_list = processor.get_labels_verification()
    num_labels = len(label_list)

    model = BertForSequenceClassification.from_pretrained(
        args.output_dir, num_labels=num_labels
    )
    tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=False)

    input_doc_file = args.dev_doc_file
    if args.dataset == "test":
        input_doc_file = args.test_doc_file

    eval_features, num_eg = get_eval_data(
        logger,
        db_path=args.db_path,
        data_dir=args.data_dir,
        output_file_name=args.predict_sentence_file_name,
        processor=processor,
        tokenizer=tokenizer,
        output_mode=output_mode,
        label_list=label_list,
        label_verification_list=label_verification_list,
        max_seq_length=args.max_seq_length,
        dataset=args.dataset,
        do_doc_process=args.do_doc_process,
        add_ro=args.add_ro,
        doc_file=input_doc_file,
        ro_doc_file=args.dev_ro_doc_file,
        api=args.api,
    )

    trainer = Trainer(model=model, args=args)
    logger.info("If predicting with onnx optimized model: {}".format(args.onnx))
    logits, _, new_guids, guids_map = trainer.predict(
        eval_features, num_eg, onnx=args.onnx
    )

    # topk selector: get dataset for nli module (dev, test)
    get_topk_sentences_eval(
        zip(new_guids, logits),
        guids_map,
        os.path.join(args.data_dir, args.predict_sentence_file_name),
        args.predict_rte_file,
        args.sent_k,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dev_fair", type=str)
    args = parser.parse_args()
    args = _get_sent_configs(args)
    LogHelper.setup()
    # can score separately
    logger = LogHelper.get_logger(os.path.splitext(os.path.basename(__file__))[0])
    predict(logger, args)
