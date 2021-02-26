"""Config parsing for NLI module"""

from configs.config import Config


def _get_nli_configs(args):
    from argparse import Namespace

    _args = Namespace()

    for k, v in Config.model_params_nli.items():
        setattr(_args, k, v)
    for k, v in Config.trainer_params.items():
        setattr(_args, k, v)

    for k, v in Config.get_args().items():
        setattr(_args, k, v)

    setattr(_args, "output_dir", Config.out_dir_rte)

    # update configs with passed args
    for k, v in (vars(args)).items():
        setattr(_args, k, v)
    return _args
