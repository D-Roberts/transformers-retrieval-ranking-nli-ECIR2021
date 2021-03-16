import argparse
import json
import os
import time
from multiprocessing.pool import ThreadPool

import wikipedia
from tqdm import tqdm

from mfactcheck.utils.dataset.data_utils import normalize
from mfactcheck.utils.dataset.reader import JSONLineReader


def processed_line(method, line):
    nps, summaries, pages = method.exact_match(line)
    line["predicted_pages"] = pages
    line["wiki_results"] = summaries  # replace with summaries for ro
    return line


def process_line_with_progress(method, line, progress=None):
    if progress is not None and line["id"] in progress:
        return progress[line["id"]]
    else:
        return processed_line(method, line)


class Ro_Doc_Retrieval:
    def __init__(self, add_claim=True, k_wiki_results=1):
        self.add_claim = add_claim
        self.k_wiki_results = k_wiki_results

    def exact_match(self, line):
        noun_phrases = line["noun_phrases"]
        page_dict = {}
        docs = []

        for np in noun_phrases:
            if len(np) >= 300:
                continue
            i = 1
            while i < 9:
                try:
                    wikipedia.set_lang("RO")
                    docs = wikipedia.search(np)
                    # print('docs are', docs)
                    for doc in docs[: self.k_wiki_results]:
                        if doc and doc not in page_dict:
                            try:
                                p = wikipedia.page(doc)
                                page_dict[doc] = p.summary
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
    method = Ro_Doc_Retrieval(add_claim=add_claim, k_wiki_results=k_wiki)
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
                    lambda l: process_line_with_progress(method, l, progress), lines
                ),
                total=len(lines),
            ):
                processed[line["id"]] = line
                progress[line["id"]] = line
                # print(line)
        with open(os.path.join(path, out_file), "w+") as f2:
            for line in lines:
                f2.write(json.dumps(processed[line["id"]]) + "\n")
    finally:
        with open(os.path.join(path, in_file + ".progress"), "wb") as f_progress:
            import pickle

            pickle.dump(progress, f_progress, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-file", type=str, help="input dataset is the retrieved En doc file"
    )
    parser.add_argument("--out-file", type=str, help="path to save Ro docs")
    parser.add_argument("--k-wiki", type=int, help="first k pages for wiki search")
    parser.add_argument("--parallel", type=bool, default=True)
    parser.add_argument("--add-claim", type=bool, default=True)
    args = parser.parse_args()

    main(args.k_wiki, args.in_file, args.out_file, args.add_claim, args.parallel)
