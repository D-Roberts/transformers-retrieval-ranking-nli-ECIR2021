"""NLI train script 1 GPU"""
import argparse
import os

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.tokenization import BertTokenizer

from mfactcheck.multi_nli.config_util import _get_nli_configs
from mfactcheck.multi_nli.data import NLIProcessor, convert_examples_to_features
from trainer import Trainer
from utils.log_helper import LogHelper


def train(logger, args):

    os.makedirs(args.output_dir, exist_ok=True)

    processor = NLIProcessor()
    output_mode = "classification"

    # # Prepare inputs
    label_list = processor.get_labels()
    num_labels = len(label_list)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=False)
    train_examples = processor.get_train_examples(args.data_dir, args.train_rte_file)
    train_examples = train_examples[0:10]  # debugging

    num_eg = len(train_examples)

    train_features = convert_examples_to_features(
        train_examples, label_list, args.max_seq_length, tokenizer, output_mode
    )

    # # Prepare model
    cache_dir = (
        args.cache_dir
        if args.cache_dir
        else os.path.join(
            str(PYTORCH_PRETRAINED_BERT_CACHE), "distributed_{}".format(args.local_rank)
        )
    )
    model = BertForSequenceClassification.from_pretrained(
        args.bert_model, cache_dir=cache_dir, num_labels=num_labels
    )

    # # Train
    trainer = Trainer(module="nli", model=model, args=args, tokenizer=tokenizer)
    trainer.train(train_features, num_labels, num_eg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    # From configs
    args = _get_nli_configs(args)
    LogHelper.setup()
    # can train separately
    logger = LogHelper.get_logger(os.path.splitext(os.path.basename(__file__))[0])
    train(logger, args)
