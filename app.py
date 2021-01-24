"""
Flask app
"""
import json
import os

from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators

from mfactcheck.multi_nli.config_util import _get_nli_configs
from mfactcheck.multi_retriever.document.api_doc_retrieval import main as doc_retrieval
from mfactcheck.multi_retriever.sentences.config_util import _get_sent_configs
from pipeline import nli, sentence_retrieval
from utils.dataset.reader import JSONLineReader


app = Flask(__name__)

cur_dir = os.path.dirname(__file__)
DATA_DIR = os.path.join(cur_dir, "data/data_dir")  # should have this dir already
PRED_DIR = os.path.join(cur_dir, "out_dir_rte")
DOC_DIR = os.path.join(cur_dir, "data/fever")


def read_pred():
    predictions_file_path = os.path.join(PRED_DIR, "refined_predictions.jsonl")
    jlr = JSONLineReader()
    lines = jlr.read(predictions_file_path)
    return lines[0]["predicted_evidence"], lines[0]["predicted_label"]


def write_claim_json(claim):
    input_list = [{"id": 1, "claim": claim}]
    claim_file_path = os.path.join(DATA_DIR, "input.jsonl")
    with open(claim_file_path, "w") as f:
        for l in input_list:
            f.write(json.dumps(l) + "\n")


# Flask
class ClaimForm(Form):
    claimsubmit = TextAreaField(
        "", [validators.DataRequired(), validators.length(min=20)]
    )


@app.route("/")
def index():
    form = ClaimForm(request.form)
    return render_template("claimform.html", form=form)


@app.route("/results", methods=["POST"])
def results():
    form = ClaimForm(request.form)
    if request.method == "POST" and form.validate():
        claim = request.form["claimsubmit"]
        write_claim_json(claim)

        app.logger.info("Starting document retrieval...")
        doc_retrieval(
            3,
            os.path.join(DATA_DIR, "input.jsonl"),
            os.path.join(DOC_DIR, "test.wiki7.jsonl"),
        )

        app.logger.info("Sentence retrieval...")

        from argparse import Namespace

        args = Namespace()
        setattr(args, "add_ro", False)
        setattr(args, "dataset", "test")
        setattr(args, "task", "predict")
        setattr(args, "onnx", True)  # should have onnx model in out_dir_sent
        setattr(args, "api", True)
        args = _get_sent_configs(args)

        sentence_retrieval(app.logger, args)

        app.logger.info("Verification step...")
        from argparse import Namespace

        args = Namespace()
        args = _get_nli_configs(args)
        setattr(args, "task", "predict")
        setattr(args, "api", True)
        nli(app.logger, args)

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
