import os
import argparse
from prettytable import PrettyTable

from fever.scorer import fever_score

from utils.dataset.reader import JSONLineReader
from configs.config import Config


def main(args):

    # load the actual evaluation set - ground truth
    jlr = JSONLineReader()
    data_lines = jlr.read(args.actual_data_file)

    predictions_file_path = os.path.join(args.out_dir_rte, "refined_predictions.jsonl")
    submission_lines = jlr.read(predictions_file_path)
    sorted_lines = []
    for g1, line in enumerate(data_lines[:10]):  # debug
        instance = {}
        instance["id"] = line["id"]
        instance.update(submission_lines[g1])
        sorted_lines.append(instance)

    score, acc, precision, recall, f1 = fever_score(
        sorted_lines, data_lines[:10]
    )  # debug
    tab = PrettyTable()
    tab.field_names = [
        "FEVER Score",
        "Label Accuracy",
        "Evidence Precision",
        "Evidence Recall",
        "Evidence F1",
    ]
    tab.add_row(
        (
            round(score, 4),
            round(acc, 4),
            round(precision, 4),
            round(recall, 4),
            round(f1, 4),
        )
    )
    print(tab)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir_rte", type=str, default=Config.out_dir_rte)
    parser.add_argument(
        "--actual_data_file", type=str, default="data/fever-data/dev.jsonl"
    )
    args = parser.parse_args()
    main(args)
