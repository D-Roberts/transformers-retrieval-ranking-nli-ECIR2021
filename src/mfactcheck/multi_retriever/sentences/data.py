import csv
import itertools
import os

import nltk
from tqdm import tqdm

from mfactcheck.utils.dataset.data_processor import DataProcessor
from mfactcheck.utils.dataset.data_utils import (
    _clean,
    clean_text,
    get_valid_texts,
    get_whole_evidence,
)
from mfactcheck.utils.dataset.fever_doc_db import FeverDocDB
from mfactcheck.utils.dataset.reader import JSONLineReader


class InputExample(object):
    """A single training example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputExample_train(object):
    """A single training example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, verification_label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.verification_label = verification_label


class InputExample_dev(object):
    """A single training example for simple sequence classification."""

    def __init__(
        self,
        guid,
        text_a,
        text_b=None,
        label=None,
        evidence_address_page=None,
        evidence_address_sent=None,
    ):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.evidence_address_page = evidence_address_page
        self.evidence_address_sent = evidence_address_sent


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class InputFeatures_train(object):
    """A single set of features of data."""

    def __init__(
        self, input_ids, input_mask, segment_ids, label_id, verification_label_id, guid
    ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.verification_label = verification_label_id
        self.guid = guid


class InputFeatures_dev(object):
    """A single set of features of data."""

    def __init__(
        self,
        input_ids,
        input_mask,
        segment_ids,
        label_id,
        evidence_address_page,
        evidence_address_sent,
        guid,
    ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.evidence_address_page = evidence_address_page
        self.evidence_address_sent = evidence_address_sent
        self.guid = guid


class SentenceProcessor(DataProcessor):
    """Processor for the sentence retrieval module."""

    def get_train_examples(
        self,
        db_path,
        data_dir,
        doc_file=None,
        num_neg_samples=None,
        do_doc_process=False,
        do_get_train_eval=False,
        tsv_file=None,
        tsv_file_neg=None,
        train_sentence_file_eval=None,
        add_ro=False,
        train_ro_doc_file=None,
        num_ro_samples=None,
    ):

        if do_doc_process:
            X_pos, X_neg = self.train_sampling(db_path, doc_file, num_neg_samples)
            # Train files from the retrieved docs
            self.write_pos_sentences(tsv_file, X_pos)
            self.write_neg_sentences(tsv_file_neg, X_neg)

        if add_ro:
            X_ro_neg = self.ro_train_sampling(train_ro_doc_file, num_ro_samples)
            self.write_neg_sentences(
                os.path.join(data_dir, "ro_train_tsv_file.tsv"), X_ro_neg
            )
            self.get_enro_train_neg(
                tsv_file_neg,
                os.path.join(data_dir, "ro_train_tsv_file.tsv"),
                tsv_file_neg,
            )
            os.remove(os.path.join(data_dir, "ro_train_tsv_file.tsv"))

        if do_get_train_eval:
            # Get file to use for scoring sentence selection on train set
            # for retrieving train_rte set
            self._get_sentence_selection_train_eval_file(
                tsv_file_neg, tsv_file, os.path.join(data_dir, train_sentence_file_eval)
            )

        train_examples_pos = self._get_train_examples(data_dir, tsv_file)
        # neg eg can be En or En+Ro
        train_examples_neg = self._get_train_examples(data_dir, tsv_file_neg)

        return train_examples_pos, train_examples_neg

    def _get_train_examples(self, data_dir, tsv_file):
        """See base class."""
        return self._create_examples_train(
            self._read_tsv(os.path.join(data_dir, tsv_file)), "train"
        )

    def get_dev_examples(
        self,
        db_path,
        data_dir,
        file_name,
        doc_file,
        dataset,
        do_doc_process=False,
        add_ro=False,
        ro_doc_file=None,
        api=False,
    ):
        """
        Sample the retrieved documents for the dev/test claims; dataset is one of
        'test', 'dev_fair', 'dev_golden'; write dataset for the records if doc_process
        otw assume doc dataset has been previously retrieved
        """
        if api:
            self.ro_dev_sampling(doc_file, data_dir, file_name, api=api)
            return self._create_examples_dev(
                self._read_tsv(os.path.join(data_dir, file_name)), "dev"
            )

        if do_doc_process:
            self._eval_sampling(db_path, doc_file, data_dir, file_name, dataset)

        if add_ro:
            self.ro_dev_sampling(ro_doc_file, data_dir, "ro_sent.tsv", api=api)
            self.get_enro_dev_dataset(
                os.path.join(data_dir, file_name),
                os.path.join(data_dir, "ro_sent.tsv"),
                os.path.join(data_dir, file_name),
            )  # overwrite

            os.remove(os.path.join(data_dir, "ro_sent.tsv"))

        return self._create_examples_dev(
            self._read_tsv(os.path.join(data_dir, file_name)), "dev"
        )

    def get_labels(self):
        """See base class."""
        return ["True", "False"]

    def get_labels_verification(self):
        """See base class."""
        return ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]

    def _write_sentences(self, file_path, x):
        with open(file_path, "wt", encoding="utf-8") as out_file:
            tsv_writer = csv.writer(out_file, delimiter="\t")
            tsv_writer.writerow(
                [
                    "id",
                    "evidence",
                    "claim",
                    "evidence_label",
                    "label",
                    "evidence_address_page",
                    "evidence_address_sent",
                ]
            )
            for line in x:
                tsv_writer.writerow(line)

    def write_neg_sentences(self, file_path, x):
        with open(file_path, "wt", encoding="utf-8") as out_file:
            tsv_writer = csv.writer(out_file, delimiter="\t")
            tsv_writer.writerow(["id", "evidence", "claim", "evidence_label", "label"])
            for line in x:
                tsv_writer.writerow(
                    [line[0], line[1], line[2], False, "NOT ENOUGH INFO"]
                )

    def write_pos_sentences(self, file_path, x):
        with open(file_path, "wt", encoding="utf-8") as out_file:
            tsv_writer = csv.writer(out_file, delimiter="\t")
            tsv_writer.writerow(["id", "evidence", "claim", "evidence_label", "label"])
            for line in x:
                if line[3] == "VERIFIABLE":
                    tsv_writer.writerow([line[0], line[1], line[2], True, line[4]])

    def _eval_sampling(self, db_path, datapath, data_dir, file_name, dataset=None):
        jlr = JSONLineReader()
        X = []
        gset = set()
        allid = set()
        with open(datapath, "r") as f:
            lines = jlr.process(f)
            print("Open document length: ", len(lines))
            for line_id, line in tqdm(enumerate(lines)):
                allid.add(line_id)
                verif_label = None if dataset == "test" else line["label"]
                s_db = FeverDocDB(db_path)
                p_lines = []
                p_titles = []
                if dataset == "test":
                    evidence_set = set()
                else:
                    evidence_set = set(
                        [
                            (evidence[2], evidence[3])
                            for evidences in line["evidence"]
                            for evidence in evidences
                        ]
                    )
                pages = [page for page in line["predicted_pages"] if page is not None]
                pages = set(pages)

                # force add all the pages from evidence set
                if dataset == "dev_golden":
                    pages = pages.union(
                        set([ev[0] for ev in evidence_set if ev[0] is not None])
                    )

                for page in pages:
                    doc_lines = s_db.get_doc_lines(page)
                    current = get_valid_texts(doc_lines, page)
                    if current:
                        p_lines.extend(current)
                        p_titles.extend([page] * len(current))

                c = 0
                # Add all sentences for dev sets.
                for title, doc_line in zip(p_titles, p_lines):
                    sample = doc_line[0]
                    if sample:
                        if (doc_line[1], doc_line[2]) not in evidence_set:
                            X.append(
                                (
                                    str(line_id) + "_" + str(c) + "_2",
                                    clean_text(" [ " + title + " ] " + sample),
                                    clean_text(line["claim"]),
                                    False,
                                    verif_label,
                                    doc_line[1],
                                    doc_line[2],
                                )
                            )
                        else:
                            X.append(
                                (
                                    str(line_id) + "_" + str(c) + "_1",
                                    clean_text(" [ " + title + " ] " + sample),
                                    clean_text(line["claim"]),
                                    True,
                                    verif_label,
                                    doc_line[1],
                                    doc_line[2],
                                )
                            )
                        gset.add(line_id)
                        c += 1
            noevid = allid.difference(gset)
            for line_id, line in enumerate(lines):
                if line_id in noevid:
                    if dataset == "test":
                        X.append(
                            (
                                str(line_id) + "_" + str(0) + "_2",
                                None,
                                clean_text(line["claim"]),
                                False,
                                verif_label,
                                None,
                                None,
                            )
                        )
                    else:
                        X.append(
                            (
                                str(line_id) + "_" + str(0) + "_2",
                                "NA",
                                clean_text(line["claim"]),
                                line["verifiable"] != "VERIFIABLE",
                                verif_label,
                                None,
                                None,
                            )
                        )
        # write dev, test datasets for the records
        self._write_sentences(os.path.join(data_dir, file_name), X)

    def train_sampling(self, db_path, datapath, num_sampling=5):
        jlr = JSONLineReader()
        X_pos, X_neg = [], []
        count = 0
        with open(datapath, "r") as f:
            lines = jlr.process(f)
            print("Open document length: ", len(lines))
            for line_id, line in tqdm(enumerate(lines)):
                count += 1
                neg_sents = []
                verif_label = line["label"]
                s_db = FeverDocDB(db_path)
                pos_set = set()
                if line["label"].upper() != "NOT ENOUGH INFO":
                    for evidence_set in line["evidence"]:
                        pos_sent = get_whole_evidence(evidence_set, s_db)
                        if pos_sent in pos_set:
                            continue
                        pos_set.add(pos_sent)
                    # get pos set for verifiable claims
                    for pos_id, pos_sent in enumerate(pos_set):
                        X_pos.append(
                            (
                                str(line_id) + "_" + str(pos_id),
                                clean_text(pos_sent),
                                clean_text(line["claim"]),
                                line["verifiable"],
                                verif_label,
                            )
                        )

                # Now find negatives for any claim
                p_lines = []
                p_titles = []
                evidence_set = set(
                    [
                        (evidence[2], evidence[3])
                        for evidences in line["evidence"]
                        for evidence in evidences
                    ]
                )
                pages = [page for page in line["predicted_pages"] if page is not None]
                for page in pages:
                    doc_lines = s_db.get_doc_lines(page)
                    current = get_valid_texts(doc_lines, page)
                    p_lines.extend(get_valid_texts(doc_lines, page))
                    p_titles.extend([page] * len(current))

                for title, doc_line in zip(p_titles, p_lines):
                    if (doc_line[1], doc_line[2]) not in evidence_set:
                        neg_sents.append((title, doc_line[0]))

                if len(neg_sents) < num_sampling:
                    num_sampling = len(neg_sents)
                if num_sampling == 0:
                    continue
                else:
                    samples = neg_sents
                    count_neg = 0
                    for neg_id, (title, sample) in enumerate(samples):
                        if sample and count_neg <= num_sampling:
                            X_neg.append(
                                (
                                    str(line_id) + "_" + str(count_neg),
                                    clean_text(" [ " + title + " ] " + sample),
                                    clean_text(line["claim"]),
                                    False,
                                    "NOT_ENOUGH_INFO",
                                )
                            )
                            count_neg += 1
        return X_pos, X_neg

    def ro_train_sampling(self, datapath, num_sample=2):
        jlr = JSONLineReader()
        X_neg = []
        with open(datapath, "r") as f:
            lines = jlr.process(f)
            print("Open document length: ", len(lines))
            for line_id, line in tqdm(enumerate(lines)):
                neg_sents = []
                claim = line["claim"]
                if line["label"].upper() != "NOT ENOUGH INFO":
                    # FOR TRAINING DATA ONLY INCLUDE VERIFIABLE CLAIMS SO THE SAME FOR RO
                    pages = line["predicted_pages"]
                    summaries = line["wiki_results"]
                    for page, summary in zip(pages, summaries):
                        # English sentence tokenizer works better empirically
                        sents = nltk.sent_tokenize(summary, language="english")
                        for sent in sents:
                            neg_sents.append((page, _clean(sent)))
                    count_neg = 5  # start count at 5 because first 5 are English
                    for neg_id, (title, sample) in enumerate(neg_sents):
                        if sample and count_neg <= num_sample + 5:
                            X_neg.append(
                                (
                                    str(line_id) + "_" + str(count_neg),
                                    " [ " + title + " ] " + sample,
                                    claim,
                                    False,
                                    "NOT_ENOUGH_INFO",
                                )
                            )
                            count_neg += 1
        return X_neg

    def ro_dev_sampling(self, datapath, data_dir, ro_file_name, api):
        jlr = JSONLineReader()
        X_neg = []
        gset = set()
        with open(datapath, "r") as f:
            lines = jlr.process(f)
            print("Open document length: ", len(lines))
            for line_id, line in tqdm(enumerate(lines)):
                neg_sents = []
                pages = line["predicted_pages"]
                label = line["label"] if not api else None
                summaries = line["wiki_results"]  # contains summaries
                for page, summary in zip(pages, summaries):
                    sents = nltk.sent_tokenize(summary, language="english")
                    for sent in sents:
                        neg_sents.append((page, _clean(sent)))
                # Use all sentences
                for neg_id, (title, sample) in enumerate(neg_sents, 1):
                    if sample:
                        X_neg.append(
                            (
                                str(line_id) + "_" + str(neg_id * 1) + "_2",
                                " [ " + title + " ] " + sample,
                                line["claim"],
                                False,
                                label,
                                None,
                                None,
                            )
                        )
                        gset.add(line_id)
        self._write_sentences(os.path.join(data_dir, ro_file_name), X_neg)

    def get_enro_train_neg(self, en_input_file, ro_input_file, en_ro_neg_tsv_path):
        with open(en_input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=None)
            lines_en = []
            for i, line in enumerate(reader):
                if i > 0:
                    lines_en.append(line)
        ro_en_neg = []
        for i, line in enumerate(lines_en):
            guid1, guid2 = line[0].split("_")
            if int(guid2) < 5:
                ro_en_neg.append(line)

        # read in the ro file same format
        with open(ro_input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=None)
            lines_ro = []
            for i, line in enumerate(reader):
                if i > 0:
                    lines_ro.append(line)

        for i, line in enumerate(lines_ro):
            ro_en_neg.append(line)

        with open(en_ro_neg_tsv_path, "wt", encoding="utf-8") as out_file:
            tsv_writer = csv.writer(out_file, delimiter="\t")
            tsv_writer.writerow(["id", "evidence", "claim", "evidence_label", "label"])
            for line in ro_en_neg:
                tsv_writer.writerow(line)

    def get_enro_dev_dataset(self, en_input_file, ro_input_file, en_ro_tsv_path):
        gset = set()
        with open(en_input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=None)
            en_ro_dev = []
            for i, line in enumerate(reader):
                if i == 0:
                    continue
                g1, g2, g3 = line[0].split("_")
                gset.add(g1)
                en_ro_dev.append(line)
        # read in the ro file same format
        with open(ro_input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=None)
            for i, line in enumerate(reader):
                if i == 0:
                    continue
                g1, g2, g3 = line[0].split("_")
                gset.add(g1)
                en_ro_dev.append(
                    [
                        line[0],
                        clean_text(line[1]),
                        clean_text(line[2]),
                        line[3],
                        line[4],
                        line[5],
                        line[6],
                    ]
                )
        with open(en_ro_tsv_path, "wt", encoding="utf-8") as out_file:
            tsv_writer = csv.writer(out_file, delimiter="\t", quotechar=None)
            tsv_writer.writerow(
                [
                    "id",
                    "evidence",
                    "claim",
                    "evidence_label",
                    "label",
                    "evidence_address_page",
                    "evidence_address_sent",
                ]
            )
            for line in en_ro_dev:
                tsv_writer.writerow(line)

    def _get_sentence_selection_train_eval_file(
        self, neg_infile, pos_infile, out_eval_file
    ):
        eval_dataset = []
        with open(neg_infile, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=None)
            for i, line in enumerate(reader):
                if i == 0:
                    continue
                eval_dataset.append([line[0] + "_2"] + line[1:])
        with open(pos_infile, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=None)
            for i, line in enumerate(reader):
                if i == 0:
                    continue
                eval_dataset.append([line[0] + "_1"] + line[1:])
        # train_eval data for retrieving rte train dataset
        with open(out_eval_file, "wt", encoding="utf-8") as out_file:
            tsv_writer = csv.writer(out_file, delimiter="\t", quotechar=None)
            tsv_writer.writerow(["id", "evidence", "claim", "evidence_label", "label"])
            for line in eval_dataset:
                tsv_writer.writerow(line)

    def _create_examples_train(self, lines, set_type):
        """Creates examples for the training set."""
        examples = []
        for i, line in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]  # evidence
            text_b = line[2]  # claim
            label = line[3]  # evidence label
            verification_label = line[4]
            examples.append(
                InputExample_train(
                    guid=guid,
                    text_a=text_a,
                    text_b=text_b,
                    label=label,
                    verification_label=verification_label,
                )
            )
        return examples

    def _create_examples_dev(self, lines, set_type):
        """Creates examples for the dev set and test set."""
        examples = []
        for i, line in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]  # evidence
            text_b = line[2]  # claim
            label_id = line[3]  # sent labels
            evidence_address_page = line[3]  # evidence_address_page
            evidence_address_sent = line[4]  # evidence_address_sentence number
            examples.append(
                InputExample_dev(
                    guid=guid,
                    text_a=text_a,
                    text_b=text_b,
                    label=label_id,
                    evidence_address_page=evidence_address_page,
                    evidence_address_sent=evidence_address_sent,
                )
            )
        return examples


def convert_examples_to_features(
    logger, examples, label_list, max_seq_length, tokenizer, output_mode
):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[: (max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        verification_label_id = label_map[example.verification_label]
        label_id = float(example.label)

        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id,
                verification_label_id=verification_label_id,
                guid=example.guid,
            )
        )
    return features


def convert_examples_to_features_train(
    logger,
    examples,
    label_list,
    label_verification_list,
    max_seq_length,
    tokenizer,
    output_mode,
):

    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}
    label_verification_map = {
        label: i for i, label in enumerate(label_verification_list)
    }

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[: (max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        verification_label_id = label_verification_map[example.verification_label]

        # if ex_index < 2:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures_train(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id,
                verification_label_id=verification_label_id,
                guid=example.guid,
            )
        )
    return features


def convert_examples_to_features_eval(
    logger,
    examples,
    label_list,
    label_verification_list,
    max_seq_length,
    tokenizer,
    output_mode,
):
    """Loads a data file into a list of `InputBatch`s."""

    # sentence True False labels
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[: (max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]

        # if ex_index < 2:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures_dev(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id,
                evidence_address_page=example.evidence_address_page,
                evidence_address_sent=example.evidence_address_sent,
                guid=example.guid,
            )
        )

    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def get_eval_data(
    logger,
    db_path,
    data_dir,
    output_file_name,
    processor,
    tokenizer,
    output_mode,
    label_list,
    label_verification_list,
    max_seq_length,
    doc_file,
    dataset,
    do_doc_process=False,
    add_ro=False,
    ro_doc_file=None,
    api=False,
):
    """For the given file processor model etc get features and labels"""
    if dataset == "train":
        eval_examples = processor._create_examples_dev(
            processor._read_tsv(os.path.join(data_dir, output_file_name)), "dev"
        )
    else:
        eval_examples = processor.get_dev_examples(
            db_path,
            data_dir,
            output_file_name,
            doc_file,
            dataset,
            do_doc_process=do_doc_process,
            add_ro=add_ro,
            ro_doc_file=ro_doc_file,
            api=api,
        )

    # eval_examples = eval_examples[0:20]  # debugging
    num_eg = len(eval_examples)
    eval_features = convert_examples_to_features_eval(
        logger,
        eval_examples,
        label_list,
        label_verification_list,
        max_seq_length,
        tokenizer,
        output_mode,
    )

    return eval_features, num_eg


def get_topk_sentences_eval(scores, guids_map, in_file_path, out_file_path, k=5):
    """Get top 5 sentences as input to the nli module"""
    g1_g2_truescore = []
    guids1 = []
    guids3 = []

    for newg, score in scores:
        line = guids_map[newg].split("-")[-1].split("_")
        g1_g2_truescore.append((line[0], line[1], line[2], score[0]))
        guids1.append(line[0])
        guids3.append(line[2])

    g1_g2_truescore.sort(key=lambda x: (x[0], -x[3]))
    final_g1_g2 = set()

    for _, g in itertools.groupby(g1_g2_truescore, lambda x: x[0]):
        for x in list(g)[:k]:  # top 5 by True score
            final_g1_g2.add(str(x[0]) + "_" + str(x[1]) + "_" + str(x[2]))

    dev_rte = []
    with open(in_file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=None)
        for i, line in enumerate(reader):
            if i > 0 and line[0] in final_g1_g2:
                dev_rte.append(line)

    with open(out_file_path, "wt") as out_file:
        tsv_writer = csv.writer(out_file, delimiter="\t")
        tsv_writer.writerow(
            [
                "id",
                "evidence",
                "claim",
                "evidence_label",
                "label",
                "evidence_address_page",
                "evidence_address_sent",
            ]
        )
        for line in dev_rte:
            label = line[4]
            if not label:  # predict input gets a mock label
                label = "NOT ENOUGH INFO"
            tsv_writer.writerow(
                [line[0], line[1], line[2], line[3], label, line[5], line[6]]
            )


def get_topk_sentences_train(
    scores,
    guids_map,
    sentence_file,
    pos_sent_file,
    neg_sent_file,
    out_file_rte_name,
    k=5,
):
    """Indices are 1/2 for pos/neg, sent id, claim id"""
    g1_g2_truescore = []
    guids1 = []
    guids3 = []

    for newg, score in scores:
        line = guids_map[newg].split("-")[-1].split("_")
        g1_g2_truescore.append((line[0], line[1], line[2], score[0]))
        guids1.append(line[0])
        guids3.append(line[2])

    g1_g2_truescore.sort(key=lambda x: (x[0], -x[3]))
    final_g1_g2 = set()
    for _, g in itertools.groupby(g1_g2_truescore, lambda x: x[0]):
        for x in list(g)[:k]:  # top 5 by True score
            final_g1_g2.add(str(x[2]) + "_" + str(x[0]) + "_" + str(x[1]))

    # Pick the top k sentences for each claim
    dev_rte = []
    with open(neg_sent_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=None)
        for line in reader:
            if "2_" + line[0] in final_g1_g2:
                dev_rte.append([line[0] + "_2"] + line[1:])

    with open(pos_sent_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=None)
        for line in reader:
            if "1_" + line[0] in final_g1_g2:
                dev_rte.append([line[0] + "_1"] + line[1:])

    with open(out_file_rte_name, "wt", encoding="utf-8") as out_file:
        tsv_writer = csv.writer(out_file, delimiter="\t")
        tsv_writer.writerow(["id", "evidence", "claim", "evidence_label", "label"])
        for line in dev_rte:
            tsv_writer.writerow(line)
