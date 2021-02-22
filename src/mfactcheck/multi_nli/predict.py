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
from trainer import Trainer
from utils.log_helper import LogHelper
from utils.model_utils import get_model_dir


def predictions_aggregator(
    logger, args, preds, labels, new_guids, guids_map, compute_acc=False
):
    """Get the refined predictions with predicted verification label and
    associated evidence and save to a json that can be scored with fever scorer.
    """

    guids1 = [int(guids_map[k].split("-")[-1].split("_")[0]) for k in new_guids]
    gset = set()
    predicted_evidence = collections.defaultdict(list)

    with open(
        os.path.join(args.data_dir, args.predict_rte_file), "r", encoding="utf-8"
    ) as f:
        reader = csv.reader(f, delimiter="\t", quotechar=None)
        for i, line in enumerate(reader):
            if i == 0:
                continue
            g1 = line[0].split("_")[0]
            gset.add(g1)
            try:
                if args.api or args.translated:
                    predicted_evidence[int(g1)].append(line[1])
                else:
                    predicted_evidence[int(g1)].append([line[5], int(line[6])])
            except Exception as e:
                continue

    # get refined preds for each g1=claim id
    refined_preds = []
    refined_labels = []
    for g1 in range(max(guids1) + 1):
        pair_preds = {preds[j] for j, x in enumerate(guids1) if x == g1}
        pair_labels = {labels[j] for j, x in enumerate(guids1) if x == g1}
        pred = 2
        if 0 in pair_preds:
            pred = 0
        if 1 in pair_preds and 0 not in pair_preds:
            pred = 1
        if len(pair_preds) > 0:
            refined_preds.append((g1, pred))

        label = 2
        if 0 in pair_labels:  # supported?
            label = 0
        if 1 in pair_labels:  # refuted?
            label = 1

        if len(pair_labels) > 0:
            refined_labels.append(label)

    # compute label accuracy
    if compute_acc:
        logger.info(
            "Label accuracy calculation for the scored file: {:.2f}".format(
                (
                    np.array([x[1] for x in refined_preds]) == np.array(refined_labels)
                ).mean()
            )
        )

    label_map = {2: "NOT ENOUGH INFO", 0: "SUPPORTS", 1: "REFUTES"}

    # add predicted evidence sets
    predictions = []
    for g1, pred_label in refined_preds:
        instance = {
            "predicted_label": label_map[pred_label],
            "predicted_evidence": predicted_evidence[int(g1)],
        }
        predictions.append(instance)

    predictions_file_path = os.path.join(args.output_dir, "refined_predictions.jsonl")
    with open(predictions_file_path, "w") as f:
        for i, l in enumerate(predictions):
            f.write(json.dumps(l) + "\n")


def predict(logger, args):
    processor = NLIProcessor()
    output_mode = "classification"

    # load/download the right model
    get_model_dir(args.output_dir, args.add_ro, "nli", args.onnx)

    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load the trained model
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

    preds, labels, new_guids, guids_map = trainer.predict(eval_data, num_eg, args)
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
    parser.add_argument("--onnx", type=bool, default=True)
    parser.add_argument("--add_ro", type=bool, default=False)
    args = parser.parse_args()
    # print('args here', args)
    args = _get_nli_configs(args)
    LogHelper.setup()
    # can score separately
    logger = LogHelper.get_logger(os.path.splitext(os.path.basename(__file__))[0])
    predict(logger, args)
