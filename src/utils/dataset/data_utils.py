"""
Data utils to be used by the data processors in each module.
"""

import re
import unicodedata


def normalize(text):
    """Resolve different type of unicode encodings. FROM DRQA."""
    return unicodedata.normalize("NFD", text)


def get_valid_texts(lines, page):
    if not lines:
        return []
    doc_lines = [
        doc_line.split("\t")[1] if len(doc_line.split("\t")[1]) > 1 else ""
        for doc_line in lines.split("\n")
    ]
    doc_lines = list(zip(doc_lines, [page] * len(doc_lines), range(len(doc_lines))))
    return doc_lines


def get_whole_evidence(evidence_set, db):
    """Evidence sets come in the form of indeces"""
    pos_sents = []
    for evidence in evidence_set:
        # print("what is the evidence in evidence_set in get_whole_evidence", evidence)
        page = evidence[2]

        doc_lines = db.get_doc_lines(page)
        # print("get doc_lines from db", doc_lines)
        doc_lines = get_valid_texts(doc_lines, page)
        for doc_line in doc_lines:
            # print("get doc_lines from db and some cleanup", doc_line)
            if doc_line[2] == evidence[3]:
                # print('check taht doc_line[2] in db is same as evidence[3] then retain pos as doc[0]', evidence[3], doc_line[0])
                pos_sents.append("[ " + page + " ] " + doc_line[0])
    pos_sent = " ".join(pos_sents)  # concatenate with space the found pos sent
    return pos_sent


def clean_text(text):
    text = re.sub(r"https?:\/\/.*[\r\n]*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\<a href", " ", text)
    text = re.sub(r"&amp;", "", text)
    text = re.sub(r'["|+&=*#$@/]', "", text)  # by Amir; DR: remove _ from here
    text = re.sub(r"\(", " ( ", text)  # by Amir
    text = re.sub(r"\)", " ) ", text)  # by Amir
    text = re.sub(r"LRB", " ( ", text)  # by Amir
    text = re.sub(r"RRB", " ) ", text)  # by Amir

    text = re.sub("LSB.*?RSB", "", text)  # by DR

    text = re.sub(r"\-", " ", text)
    text = re.sub(r"\_", " ", text)
    text = re.sub(r"<br />", " ", text)
    text = text.replace("...", " ")
    return text


# used in Ro docs
def remove_diacritics(text):
    """
    Returns a string with all diacritics (aka non-spacing marks) removed.
    For example "Héllô" will become "Hello".
    Useful for comparing strings in an accent-insensitive fashion.
    """
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalized if unicodedata.category(c) != "Mn")


# Used in Ro docs
def _clean(s):
    s = s.replace("\\", " ").replace("\n", " ").replace("\t", " ")
    s = remove_diacritics(s)
    return s
