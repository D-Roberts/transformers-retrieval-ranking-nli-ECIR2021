"""Config parsing for NLI module"""

from configs.config import Config


def _get_nli_configs(args):
    from argparse import Namespace

    _args = Namespace()

    for k, v in Config.model_params_nli.items():
        setattr(_args, k, v)
    for k, v in Config.trainer_params.items():
        setattr(_args, k, v)

    setattr(_args, "api", False)
    setattr(_args, "onnx", Config.onnx)
    setattr(_args, "add_ro", Config.add_ro)
    setattr(_args, "cache_dir", "")
    setattr(_args, "data_dir", Config.data_dir)
    setattr(_args, "output_dir", Config.out_dir_rte)
    setattr(_args, "train_rte_file", Config.train_rte_file)
    setattr(_args, "predict_rte_file", Config.predict_rte_file)

    # update configs with passed args
    for k, v in (vars(args)).items():
        setattr(_args, k, v)
    return _args
