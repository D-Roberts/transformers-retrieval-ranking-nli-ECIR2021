import os

from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.tokenization import BertTokenizer

from mfactcheck.utils.model_utils import get_model_dir

class MBert:
    def __init__(self, output_dir=None, num_labels=2):
        self.num_labels = num_labels
        self.load_model(output_dir)

    def load_model(self, output_dir):
        self.model = BertForSequenceClassification.from_pretrained(
            output_dir, num_labels=self.num_labels)
        self.tokenizer = BertTokenizer.from_pretrained(output_dir, do_lower_case=False)
        
    