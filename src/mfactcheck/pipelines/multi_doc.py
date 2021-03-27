import argparse
import json
import itertools
import os
import time
import unicodedata
import concurrent.futures

import wikipedia
from allennlp.predictors.predictor import Predictor
from tqdm import tqdm

from mfactcheck.utils.dataset.reader import JSONLineReader
from mfactcheck.utils.dataset.data_utils import page_clean
from mfactcheck.utils.log_helper import LogHelper

LogHelper.setup()
logger = LogHelper.get_logger(os.path.splitext(os.path.basename(__file__))[0])


def processed_line(method, line):
    nps, summaries, pages = method.exact_match(line)
    line["noun_phrases"] = nps
    line["predicted_pages"] = pages
    line["wiki_results"] = summaries
    return line


class Doc_Retrieval:
    def __init__(self, add_claim=True, k_wiki_results=1):
        self.add_claim = add_claim
        self.k_wiki_results = k_wiki_results
        self.predictor = Predictor.from_path(
            "https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz"
        )

    def get_NP(self, tree, nps):
        if isinstance(tree, dict):
            if "children" not in tree:
                if tree["nodeType"] == "NP":
                    nps.append(tree["word"])
            elif "children" in tree:
                if tree["nodeType"] == "NP":
                    nps.append(tree["word"])
                    self.get_NP(tree["children"], nps)
                else:
                    self.get_NP(tree["children"], nps)
        elif isinstance(tree, list):
            for sub_tree in tree:
                self.get_NP(sub_tree, nps)

        return nps

    def get_subjects(self, tree):
        subject_words = []
        subjects = []
        for subtree in tree["children"]:
            if (
                subtree["nodeType"] == "VP"
                or subtree["nodeType"] == "S"
                or subtree["nodeType"] == "VBZ"
            ):
                subjects.append(" ".join(subject_words))
                subject_words.append(subtree["word"])
            else:
                subject_words.append(subtree["word"])
        return subjects

    def get_noun_phrases(self, line):
        claim = line["claim"]
        tokens = self.predictor.predict(claim)
        nps = []
        tree = tokens["hierplane_tree"]["root"]
        noun_phrases = self.get_NP(tree, nps)
        subjects = self.get_subjects(tree)
        for subject in subjects:
            if len(subject) > 0:
                noun_phrases.append(subject)
        if self.add_claim:
            noun_phrases.append(claim)

        return [x for x in set(noun_phrases) if len(x) < 300]

    def searches(self, np, lang):
        page_dict = {}
        i = 1
        while i < 9:
            try:
                wikipedia.set_lang(lang)
                docs = wikipedia.search(np)
                for doc in docs[: self.k_wiki_results]:
                    if doc and lang + " " + doc not in page_dict:
                        try:
                            p = wikipedia.page(doc)
                            page_dict[lang + " " + doc] = p.summary
                        except Exception as e:
                            continue
            except (
                ConnectionResetError,
                ConnectionError,
                ConnectionAbortedError,
                ConnectionRefusedError,
            ):
                print("Connection reset error received! Trial #" + str(i))
                time.sleep(600 * i)
                i += 1
            else:
                break
        return page_dict

    def exact_match(self, line):
        noun_phrases = self.get_noun_phrases(line)
        # logger.info(f"line{line}")
        page_dict = {}
        langs = ["en", "ro", "pt"]

        combos = list(itertools.product(noun_phrases, langs))
        # print(combos)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            res = (executor.submit(self.searches, np, lang) for (np, lang) in combos)
            for future in concurrent.futures.as_completed(res):
                try:
                    r = future.result()
                    # logger.info(f"r is {r}")
                except Exception as e:
                    logger.info("some e")
                else:
                    page_dict.update(r)
        # print(page_dict)
        return noun_phrases, [*page_dict.values()], [*page_dict]


def main(doc_retriever, in_file, out_file, k_wiki, add_claim, parallel):
    method = doc_retriever
    path = os.getcwd()
    jlr = JSONLineReader()
    lines = jlr.read(os.path.join(path, in_file))
    processed = {line["id"]: processed_line(method, line) for line in lines}

    with open(os.path.join(path, out_file), "w+") as f2:
        for line in lines:
            f2.write(json.dumps(processed[line["id"]]) + "\n")
