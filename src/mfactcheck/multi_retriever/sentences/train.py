"""Train Sentence Selector 1 GPU"""

import argparse
import os

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.tokenization import BertTokenizer

from mfactcheck.multi_retriever.sentences.config_util import _get_sent_configs
from mfactcheck.multi_retriever.sentences.data import (
    SentenceProcessor,
    convert_examples_to_features_train,
    get_eval_data,
    get_topk_sentences_train,
)
from trainer import Trainer
from utils.log_helper import LogHelper


def train(logger, args):

    processor = SentenceProcessor()
    output_mode = "classification"

    # Dir for artifacts; makes fresh with each train
    os.makedirs(args.output_dir, exist_ok=True)

    label_list = processor.get_labels()
    label_verification_list = processor.get_labels_verification()
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=False)

    # also gets train_eval to score for nli train sentence selection
    train_examples_pos, train_examples_neg = processor.get_train_examples(
        db_path=args.db_path,
        data_dir=args.data_dir,
        doc_file=args.training_doc_file,
        num_neg_samples=args.num_neg_samples,
        do_doc_process=args.do_doc_process,
        do_get_train_eval=True,
        tsv_file=args.train_tsv_file_pos,
        tsv_file_neg=args.train_tsv_file_neg,
        train_sentence_file_eval=args.train_sentence_file_eval,
        add_ro=args.add_ro,
        train_ro_doc_file=args.train_ro_doc_file,
        num_ro_samples=args.num_ro_samples,
    )

    train_examples_pos = train_examples_pos[0:20]  # debugging
    train_examples_neg = train_examples_neg[0:20]  # debugging

    num_eg = len(train_examples_pos)
    num_eg_neg = len(train_examples_neg)

    train_features = convert_examples_to_features_train(
        logger,
        train_examples_pos,
        label_list,
        label_verification_list,
        args.max_seq_length,
        tokenizer,
        output_mode,
    )
    train_features_neg = convert_examples_to_features_train(
        logger,
        train_examples_neg,
        label_list,
        label_verification_list,
        args.max_seq_length,
        tokenizer,
        output_mode,
    )

    # Prepare model
    cache_dir = (
        args.cache_dir
        if args.cache_dir
        else os.path.join(
            str(PYTORCH_PRETRAINED_BERT_CACHE), f"distributed_{args.local_rank}"
        )
    )
    model = BertForSequenceClassification.from_pretrained(
        args.bert_model, cache_dir=cache_dir, num_labels=num_labels
    )
    # Trainer
    trainer = Trainer(model=model, module="sentence", args=args, tokenizer=tokenizer)
    trainer.train(
        train_data=train_features,
        num_labels=num_labels,
        num_eg=num_eg,
        num_eg_neg=num_eg_neg,
        negative_train_data=train_features_neg,
    )

    # Score train datasets and obtain train dataset for the nli module; en only or enro sentences
    train_eval_features, num_eg = get_eval_data(
        logger,
        db_path=args.db_path,
        data_dir=args.data_dir,
        output_file_name=args.train_sentence_file_eval,
        processor=processor,
        tokenizer=tokenizer,
        output_mode=output_mode,
        label_list=label_list,
        label_verification_list=label_verification_list,
        max_seq_length=args.max_seq_length,
        doc_file=None,
        dataset=args.dataset,
    )
    # Using the model we just trained
    logits, _, new_guids, guids_map = trainer.predict(train_eval_features, num_eg)

    # Top k(5) sentences will be used to train the nli module
    get_topk_sentences_train(
        zip(new_guids, logits),
        guids_map,
        args.train_sentence_file_eval,
        args.train_tsv_file_pos,
        args.train_tsv_file_neg,
        args.train_rte_file,
        args.sent_k,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="train", type=str)
    args = parser.parse_args()
    args = _get_sent_configs(args)
    LogHelper.setup()
    # can train separately
    logger = LogHelper.get_logger(os.path.splitext(os.path.basename(__file__))[0])
    train(logger, args)
