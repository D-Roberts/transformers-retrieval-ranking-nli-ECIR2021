import os

from mfactcheck.utils.log_helper import LogHelper
from mfactcheck.multi_retriever.sentences.data import SentProcessor
from .mbert import MBert

LogHelper.setup()
logger = LogHelper.get_logger(os.path.splitext(os.path.basename(__file__))[0])


class SentMBert(MBert):
    def __init__(self, output_dir, module="sent", add_ro=False):
        super().__init__(
            output_dir=output_dir, module=module, num_labels=2, add_ro=add_ro
        )
        self.processor = SentProcessor()
        self.label_list = self.processor.get_labels()
        self.num_labels = len(self.label_list)
        self.label_verification_list = self.processor.get_labels_verification()
