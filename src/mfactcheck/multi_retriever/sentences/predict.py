"""Sentence Selector Predict/Eval CPU or 1 GPU"""

import argparse
import os

from mfactcheck.multi_retriever.sentences.config_util import _get_sent_configs
from mfactcheck.multi_retriever.sentences.data import (
    SentenceProcessor,
    get_eval_data,
    get_topk_sentences_eval,
)
from mfactcheck.models.sent_mbert import SentMBert
from mfactcheck.trainer import Trainer
from mfactcheck.utils.log_helper import LogHelper


def predict(logger, args):

    module = SentMBert(args.output_dir)
    label_list = module.processor.get_labels()
    label_verification_list = module.processor.get_labels_verification()
    model = module.model

    input_doc_file = args.dev_doc_file
    if args.dataset == "test":
        input_doc_file = args.test_doc_file

    eval_features, num_eg = get_eval_data(
        logger,
        db_path=args.db_path,
        data_dir=args.data_dir,
        output_file_name=args.predict_sentence_file_name,
        processor=module.processor,
        tokenizer=module.tokenizer,
        output_mode="classification",
        label_list=module.label_list,
        label_verification_list=module.label_verification_list,
        max_seq_length=args.max_seq_length,
        dataset=args.dataset,
        do_doc_process=args.do_doc_process,
        add_ro=args.add_ro,
        doc_file=input_doc_file,
        ro_doc_file=args.dev_ro_doc_file,
        api=args.api,
    )

    trainer = Trainer(model=model, args=args)
    logits, _, new_guids, guids_map = trainer.predict(eval_features, num_eg)

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
    parser.add_argument("--add_ro", default=False, type=bool)
    args = parser.parse_args()
    args = _get_sent_configs(args)
    LogHelper.setup()
    # can score separately
    logger = LogHelper.get_logger(os.path.splitext(os.path.basename(__file__))[0])
    predict(logger, args)
