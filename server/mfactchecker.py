import csv
import json
import os
import time

from mfactcheck.configs.config import Config
from mfactcheck.utils.dataset.reader import JSONLineReader
from mfactcheck.pipelines.multi_doc import main as doc_retrieval
from mfactcheck.utils.log_helper import LogHelper

LogHelper.setup()
logger = LogHelper.get_logger(os.path.splitext(os.path.basename(__file__))[0])


class MFactChecker:
    def __init__(self, doc_retriever, sentence_selector, verifier, r):
        self.doc_retriever = doc_retriever
        self.sentence_selector = sentence_selector
        self.verifier = verifier
        # the Redis cache is r
        self.r = r
        self.post_init()

    def post_init(self):
        os.makedirs(Config.dataset_folder, exist_ok=True)
        os.makedirs(Config.data_dir, exist_ok=True)

    def predict(self):
        self.verifier()

    def get_evidence(self):

        logger.info("Starting document retrieval...")
        st = time.time()
        doc_retrieval(
            doc_retriever=self.doc_retriever,
            in_file=os.path.join(Config.data_dir, "input.jsonl"),
            out_file=os.path.join(Config.test_doc_file),
            k_wiki=1,
            add_claim=True,
            parallel=False,
        )

        logger.info(
            f"Doc retrieved in {time.time() - st}. Now sentence selection step..."
        )
        self.sentence_selector()

    def handle_claim_or_id(self, claim_id_or_claim):
        # a claim was entered
        st = time.time()
        if len(claim_id_or_claim) > 20:
            logger.info("Starting MFactChecker...")
            claim = claim_id_or_claim
            if len(self.r.keys()) == 0:
                claim_id = 1
            else:
                claim_id = max([int(x.decode("utf-8")) for x in self.r.keys()]) + 1

        else:
            # an ID was entered
            claim_id = claim_id_or_claim
            if self.r.exists(str(claim_id)):
                logger.info(
                    "Claim id found in cache, reading 5 evidence sentences for verifier scoring..."
                )
                claim = self.read_cache(claim_id)
            else:
                logger.info(
                    "And id was givenn but id not found in cache, user must re-enter..."
                )
                return (None, None, None)

        self.write_claim_json(claim, claim_id)

        if not self.r.exists(str(claim_id)):

            # get full evidence
            self.get_evidence()

            logger.info("Caching results...")
            self.write_cache(claim_id)

        logger.info("Verification step...")

        self.predict()

        evidence, label = self.read_pred()
        logger.info(f"Mfactchecker took {time.time() - st} seconds.")

        return claim, evidence, label

    def write_claim_json(self, claim, input_id):
        input_list = [{"id": input_id, "claim": claim}]
        claim_file_path = os.path.join(Config.data_dir, "input.jsonl")

        with open(claim_file_path, "w+") as f:
            for line in input_list:
                f.write(json.dumps(line) + "\n")

    def read_pred(self):
        predictions_file_path = os.path.join(
            Config.out_dir_rte, "refined_predictions.jsonl"
        )
        jlr = JSONLineReader()
        lines = jlr.read(predictions_file_path)

        return lines[0]["predicted_evidence"], lines[0]["predicted_label"]

    def read_cache(self, claim_id):
        dict_obj = json.loads(self.r.get(str(claim_id)))
        with open(Config.predict_rte_file, "w", encoding="utf-8") as f:
            tsv_writer = csv.writer(f, delimiter="\t")
            tsv_writer.writerow(["id", "evidence", "claim", "evidence_label", "label"])
            for sent in dict_obj["evidence"]:
                line = (
                    [claim_id]
                    + [sent]
                    + [dict_obj["claim"]]
                    + [False]
                    + ["NOT ENOUGH INFO"]
                )
                tsv_writer.writerow(line)
        return dict_obj["claim"]

    def write_cache(self, claim_id):
        d_to_write = {}
        d_to_write["id"] = str(claim_id)
        d_to_write["evidence"] = []

        with open(Config.predict_rte_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for i, line in enumerate(reader):
                if i == 0:
                    continue
                if i == 1:
                    d_to_write["claim"] = line[2]
                d_to_write["evidence"].append(line[1])
        json_obj = json.dumps(d_to_write)
        self.r.set(str(claim_id), json_obj)
       
