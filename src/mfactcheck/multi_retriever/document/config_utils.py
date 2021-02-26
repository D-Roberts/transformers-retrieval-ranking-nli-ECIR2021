"""Parse configs for the Document Retriever module"""

from configs.config import Config


def _get_doc_configs(args):
    from argparse import Namespace

    _args = Namespace()
    for k, v in Config.get_args().items():
        setattr(_args, k, v)

    # update configs with passed args
    for k, v in (vars(args)).items():
        setattr(_args, k, v)
    return _args
