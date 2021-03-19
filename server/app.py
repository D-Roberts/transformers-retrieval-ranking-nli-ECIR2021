#!/usr/bin/env python
# coding=utf-8
"""Flask app"""
import csv
import json
import os

from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
from redis import Redis

from mfactchecker import MFactChecker
from mfactcheck.pipelines.multi_nli import MultiNLIPipeline


app = Flask(__name__)
cur_dir = os.path.dirname(__file__)

# start redis cache for retrieved sentences

# r = Redis(host='redis', port=6379)
r = Redis(port=6379)

# load pipeline
verifier = MultiNLIPipeline()
predictor = MFactChecker(verifier, r)


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

        claim, evidence, label = predictor.handle_claim_or_id(app.logger, claim_id_or_claim)

        if claim is None:
            return render_template("unknown_id.html", form=form)

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
