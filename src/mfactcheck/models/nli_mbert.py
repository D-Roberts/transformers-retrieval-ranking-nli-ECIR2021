import os

from mfactcheck.utils.log_helper import LogHelper
from mfactcheck.multi_nli.data import NLIProcessor
from .mbert import MBert

LogHelper.setup()
logger = LogHelper.get_logger(os.path.splitext(os.path.basename(__file__))[0])


class NLIMBert(MBert):
    def __init__(self, output_dir, module="nli", add_ro=False):
        super().__init__(
            output_dir=output_dir, module=module, num_labels=3, add_ro=add_ro
        )
        self.processor = NLIProcessor()
        self.label_list = self.processor.get_labels()
        self.num_labels = len(self.label_list)
