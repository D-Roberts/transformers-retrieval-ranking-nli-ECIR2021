import time
import os

# from flask import Flask, render_template, request
# from wtforms import Form, TextAreaField, validators
# from redis import Redis

# from . import MFactChecker
# from mfactchecker import MFactChecker
from mfactcheck.pipelines.multi_doc import Doc_Retrieval
from mfactcheck.pipelines.multi_sent import MultiSentPipeline
from mfactcheck.pipelines.multi_nli import MultiNLIPipeline


cur_dir = os.path.dirname(__file__)

# load pipeline models to assemble MFactChecker
# TODO: extra copy so need refactored
doc_retriever = Doc_Retrieval(add_claim=True, k_wiki_results=1)
sentence_selector = MultiSentPipeline()
verifier = MultiNLIPipeline()
