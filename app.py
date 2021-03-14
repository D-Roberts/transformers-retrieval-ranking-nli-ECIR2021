"""
Flask app
"""
import csv
import json
import os

from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
from redis import Redis

from mfactcheck.multi_nli.config_util import _get_nli_configs
from mfactcheck.multi_retriever.document.api_doc_retrieval import main as doc_retrieval
from mfactcheck.multi_retriever.sentences.config_util import _get_sent_configs
from pipeline import nli, sentence_retrieval
from utils.dataset.reader import JSONLineReader
from configs.config import Config


app = Flask(__name__)

cur_dir = os.path.dirname(__file__)

r = Redis(host='redis', port=6379)
# r = Redis(port=6379)


def write_cache(claim_id):
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
    r.bgsave()


def read_cache(claim_id):
    dict_obj = json.loads(r.get(str(claim_id)))
    with open(Config.predict_rte_file, 'w', encoding="utf-8") as f:
        tsv_writer = csv.writer(f, delimiter = '\t')
        tsv_writer.writerow(["id", "evidence", "claim", "evidence_label", "label"])
        for sent in dict_obj['evidence']:
            line = [claim_id] + [sent] + [dict_obj['claim']] + [False] + ["NOT ENOUGH INFO"]
            tsv_writer.writerow(line)
    return dict_obj['claim']


def read_pred():
    predictions_file_path = os.path.join(
        Config.out_dir_rte, "refined_predictions.jsonl"
    )
    jlr = JSONLineReader()
    lines = jlr.read(predictions_file_path)
  
    return lines[0]["predicted_evidence"], lines[0]["predicted_label"]


def write_claim_json(claim, input_id):
    
    input_list = [{"id": input_id, "claim": claim}]
    claim_file_path = os.path.join(Config.data_dir, "input.jsonl")

    with open(claim_file_path, "w+") as f:
        for line in input_list:
            f.write(json.dumps(line) + "\n")


def run_document_retrieval():
    
    doc_path = Config.test_doc_file

    doc_retrieval(
        3,
        os.path.join(Config.data_dir, "input.jsonl"),
        os.path.join(doc_path),
    )


def run_evidence_recommendation():
    from argparse import Namespace

    args = Namespace()
    args = _get_sent_configs(args)
    setattr(args, "api", True)
    sentence_retrieval(app.logger, args)


def run_verification():
    from argparse import Namespace
    args = Namespace()
    args = _get_nli_configs(args)
    setattr(args, "api", True)
    nli(app.logger, args)


# Flask
class ClaimForm(Form):
    claimsubmit = TextAreaField(
        "", [validators.DataRequired(), validators.length(min=1)]
    )


@app.route("/")
def index():
    form = ClaimForm(request.form)
    return render_template("claimform.html", form=form)


@app.route("/results", methods=["POST"])
def results():
    form = ClaimForm(request.form)
    if request.method == "POST" and form.validate():
        claim_id_or_claim = request.form["claimsubmit"]

        if not os.path.isdir(Config.dataset_folder):
            os.makedirs(Config.dataset_folder)
        if not os.path.isdir(Config.data_dir):
            os.makedirs(Config.data_dir)

        # A claim was entered
        if len(claim_id_or_claim) > 20:
            claim = claim_id_or_claim
            if len(r.keys()) == 0:
                claim_id = 1
            else:
                claim_id = max([int(x.decode("utf8")) for x in r.keys()]) + 1
            
        else:
            # an ID was entered
            # TODO: if id does not exist in REdis - reload front page
            claim_id = claim_id_or_claim
            claim = read_cache(claim_id)

        write_claim_json(claim, claim_id)

        if not r.exists(str(claim_id)):

            app.logger.info("Starting document retrieval...")
            run_document_retrieval()

            app.logger.info("Sentence selection step...")
            run_evidence_recommendation()

            app.logger.info("Caching results...")
            write_cache(claim_id)

        app.logger.info("Verification step...")
        run_verification()

        app.logger.info("Predicting complete.")
        evidence, label = read_pred()

        return render_template(
            "results.html", content=claim, retrieved=evidence, predicted=label
        )

    return render_template("claimform.html", form=form)


@app.route("/thanks", methods=["POST"])
def feedback():
    # mock feedback
    feedback = request.form["feedback_button"]
    claim = request.form["claim"]
    retrieved = request.form["retrieved"]

    return render_template("thanks.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, processes=1)
