"""
Data utils to be used by the data processors in each module.
"""

import re
import unicodedata


def normalize(text):
    """Resolve different type of unicode encodings."""
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
        page = evidence[2]

        doc_lines = db.get_doc_lines(page)
        doc_lines = get_valid_texts(doc_lines, page)
        for doc_line in doc_lines:
            if doc_line[2] == evidence[3]:
                pos_sents.append("[ " + page + " ] " + doc_line[0])
    pos_sent = " ".join(pos_sents)
    return pos_sent


def clean_text(text):
    text = re.sub(r"https?:\/\/.*[\r\n]*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\<a href", " ", text)
    text = re.sub(r"&amp;", "", text)
    text = re.sub(r'["|+&=*#$@/]', "", text)
    text = re.sub(r"\(", " ( ", text)
    text = re.sub(r"\)", " ) ", text)
    text = re.sub(r"LRB", " ( ", text)
    text = re.sub(r"RRB", " ) ", text)

    text = re.sub("LSB.*?RSB", "", text)

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


# Used at the end
def _clean_last(page):
    page = normalize(page)
    page = page.replace("_", " ")
    page = page.replace("-LRB-", "(")
    page = page.replace("-RRB-", ")")
    page = page.replace("-COLON-", ":")
    page = page.replace("\\u200", " ")
    page = page.replace('"""', " ")
    return page


# page clean
def page_clean(page):
    page = normalize(page)
    page = page.replace(" ", "_")
    page = page.replace("(", "-LRB-")
    page = page.replace(")", "-RRB-")
    page = page.replace(":", "-COLON-")
    return page
