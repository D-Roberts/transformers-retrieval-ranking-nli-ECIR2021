"""Parse configs for the Sentence Selector module"""

from mfactcheck.configs.config import Config


def _get_sent_configs(args):
    from argparse import Namespace

    _args = Namespace()
    for k, v in Config.model_params_sentence.items():
        setattr(_args, k, v)
    for k, v in Config.trainer_params.items():
        setattr(_args, k, v)
    for k, v in Config.sent_param.items():
        setattr(_args, k, v)
    for k, v in Config.get_args().items():
        setattr(_args, k, v)

    setattr(_args, "output_dir", Config.out_dir_sent)
    setattr(_args, "train_sentence_file_eval", "train_sent_file_eval.tsv")
    setattr(_args, "predict_sentence_file_name", "predict_sent_file.tsv")

    # update configs with passed args
    for k, v in (vars(args)).items():
        setattr(_args, k, v)
    return _args
