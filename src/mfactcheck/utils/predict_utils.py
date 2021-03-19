import collections
import csv
import json 
import numpy as np
import os

from mfactcheck.utils.dataset.data_utils import _clean_last as clean


def predictions_aggregator(
        logger, args, preds, labels, new_guids, guids_map, compute_acc
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
                        predicted_evidence[int(g1)].append([clean(line[5]), int(line[6])])
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
                f"Label accuracy calculation for the scored file: \
                {(np.array([x[1] for x in refined_preds]) == np.array(refined_labels)).mean():.2f}"
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