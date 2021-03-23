import os

from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.tokenization import BertTokenizer

from mfactcheck.utils.model_utils import get_model_dir
from mfactcheck.utils.log_helper import LogHelper

LogHelper.setup()
logger = LogHelper.get_logger(os.path.splitext(os.path.basename(__file__))[0])


class MBert:
    def __init__(self, output_dir=None, module=None, num_labels=None, add_ro=False):
        self.num_labels = num_labels
        self.load_model(output_dir, add_ro, module)

    def load_model(self, output_dir, add_ro, module):
        logger.info(f"Loading {module} module model.")
        if not os.path.isdir(output_dir):
            get_model_dir(output_dir, add_ro=add_ro, module=module, onnx=False)
        self.model = BertForSequenceClassification.from_pretrained(
            output_dir, num_labels=self.num_labels)
        self.tokenizer = BertTokenizer.from_pretrained(output_dir, do_lower_case=False)
        
    