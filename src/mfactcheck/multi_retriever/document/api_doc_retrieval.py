import argparse
import json
import os
import time
import unicodedata
from multiprocessing.pool import ThreadPool

import wikipedia
from allennlp.predictors.predictor import Predictor
from tqdm import tqdm

from utils.dataset.reader import JSONLineReader


def normalize(text):
    """Resolve different type of unicode encodings. FROM DRQA."""
    return unicodedata.normalize("NFD", text)


def processed_line(method, line):
    nps, summaries, pages = method.exact_match(line)
    line["noun_phrases"] = nps
    line["predicted_pages"] = pages
    line["wiki_results"] = summaries
    return line


def process_line_with_progress(method, line, progress=None):
    if progress is not None and line["id"] in progress:
        return progress[line["id"]]
    else:
        return processed_line(method, line)


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

        return list(set(noun_phrases))

    def exact_match(self, line):
        noun_phrases = self.get_noun_phrases(line)
        page_dict = {}
        docs = []
        langs = ["en", "ro", "pt"]

        for np in noun_phrases:
            if len(np) >= 300:
                continue
            for lang in langs:
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
                                except Exception:
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

        processed_pages = []

        for i, page in enumerate(page_dict.keys()):
            page = page.replace(" ", "_")
            page = page.replace("(", "-LRB-")
            page = page.replace(")", "-RRB-")
            page = page.replace(":", "-COLON-")
            page = normalize(page)
            if ":" in page:
                page = page.replace(":", "-COLON-")
            processed_pages.append(page)
        return noun_phrases, list(page_dict.values()), processed_pages


def get_map_function(parallel, p=None):
    assert (
        not parallel or p is not None
    ), "A ThreadPool object should be given if parallel is True"
    return p.imap_unordered if parallel else map


def main(k_wiki, in_file, out_file, add_claim=True, parallel=True):
    method = Doc_Retrieval(add_claim=add_claim, k_wiki_results=k_wiki)
    processed = dict()
    path = os.getcwd()
    jlr = JSONLineReader()
    lines = jlr.read(os.path.join(path, in_file))
    if os.path.isfile(os.path.join(path, in_file + ".progress")):
        with open(os.path.join(path, in_file + ".progress"), "rb") as f_progress:
            import pickle

            progress = pickle.load(f_progress)
            print(
                os.path.join(path, in_file + ".progress")
                + " exists. Load it as progress file."
            )
    else:
        progress = dict()

    try:
        with ThreadPool(processes=4 if parallel else None) as p:
            for line in tqdm(
                get_map_function(parallel, p)(
                    lambda l: process_line_with_progress(method, l, progress=None),
                    lines,
                ),
                total=len(lines),
            ):
                processed[line["id"]] = line
                # time.sleep(0.5)
        with open(os.path.join(path, out_file), "w+") as f2:
            for line in lines:
                f2.write(json.dumps(processed[line["id"]]) + "\n")
    finally:
        print("...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-file", type=str, help="input dataset", default="data/data_dir/input.jsonl"
    )
    parser.add_argument(
        "--out-file",
        type=str,
        help="path to save output dataset",
        default="data/data_dir/en_ro_pt_docs.jsonl",
    )
    parser.add_argument(
        "--k-wiki", type=int, help="first k pages for wiki search", default=3
    )
    parser.add_argument("--parallel", type=bool, default=True)
    parser.add_argument("--add-claim", type=bool, default=True)
    args = parser.parse_args()

    main(args.k_wiki, args.in_file, args.out_file, args.add_claim, args.parallel)
