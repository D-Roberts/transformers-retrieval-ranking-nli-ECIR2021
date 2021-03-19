import csv
import json
import os 

from mfactcheck.configs.config import Config
from mfactcheck.utils.dataset.reader import JSONLineReader
from app_utils import run_document_retrieval, run_evidence_recommendation

class MFactChecker:
    def __init__(self, verifier, r):
        self.verifier = verifier
        self.r = r
        self.post_init()

    def post_init(self):
        if not os.path.isdir(Config.dataset_folder):
            os.makedirs(Config.dataset_folder)
        if not os.path.isdir(Config.data_dir):
            os.makedirs(Config.data_dir)

    def predict(self):
        self.verifier()

    def handle_claim_or_id(self, logger, claim_id_or_claim):
        # a claim was entered
        if len(claim_id_or_claim) > 20:
            logger.info("Starting full pipeline...")
            claim = claim_id_or_claim
            if len(self.r.keys()) == 0:
                claim_id = 1
            else:
                claim_id = max([int(x.decode("utf8")) for x in self.r.keys()]) + 1
            
        else:
            # an ID was entered
            claim_id = claim_id_or_claim
            if self.r.exists(str(claim_id)): 
                logger.info("Claim id found in cache, reading 5 sentences for scoring...")
                claim = self.read_cache(claim_id, self.r)
            else:
                logger.info("Claim id not found in cache, user must re-enter...")
                return (None, None, None)

        self.write_claim_json(claim, claim_id)

        if not self.r.exists(str(claim_id)):

            logger.info("Starting document retrieval...")
            run_document_retrieval()

            logger.info("Sentence selection step...")
            run_evidence_recommendation(logger)

            logger.info("Caching results...")
            self.write_cache(claim_id, self.r)

        logger.info("Verification step...")
        self.predict()

        logger.info("Predicting complete...")
        evidence, label = self.read_pred()

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

        
    def read_cache(self, claim_id, r):
        dict_obj = json.loads(r.get(str(claim_id)))
        with open(Config.predict_rte_file, 'w', encoding="utf-8") as f:
            tsv_writer = csv.writer(f, delimiter = '\t')
            tsv_writer.writerow(["id", "evidence", "claim", "evidence_label", "label"])
            for sent in dict_obj['evidence']:
                line = [claim_id] + [sent] + [dict_obj['claim']] + [False] + ["NOT ENOUGH INFO"]
                tsv_writer.writerow(line)
        return dict_obj['claim']


    def write_cache(self, claim_id, r):
        d_to_write = {}
        d_to_write["id"] = str(claim_id)
        d_to_write["evidence"] = []

        with open(Config.predict_rte_file, 'r', encoding="utf-8") as f:
            reader = csv.reader(f, delimiter='\t')
            for i, line in enumerate(reader):
                if i == 0:
                    continue
                if i == 1:
                    d_to_write["claim"] = line[2]
                d_to_write["evidence"].append(line[1])

        json_obj = json.dumps(d_to_write)
        r.set(str(claim_id), json_obj)
        # snapshot
        r.bgsave()
