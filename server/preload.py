import time
import os

from mfactcheck.pipelines.multi_doc import Doc_Retrieval
from mfactcheck.pipelines.multi_sent import MultiSentPipeline
from mfactcheck.pipelines.multi_nli import MultiNLIPipeline


cur_dir = os.path.dirname(__file__)

# simple caching of models before workers handle requests

doc_retriever = Doc_Retrieval(add_claim=True, k_wiki_results=1)
sentence_selector = MultiSentPipeline()
verifier = MultiNLIPipeline()
