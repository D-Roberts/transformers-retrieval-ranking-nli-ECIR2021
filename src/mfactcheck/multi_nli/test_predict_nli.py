# test_predict_nli_pipe
import argparse

from mfactcheck.pipelines import MultiNLIPipeline

pred_file = 'roro0_dev.tsv'
parser = argparse.ArgumentParser()
parser.add_argument(
        "--predict-rte-file",
        type=str,
        default=pred_file,
        help="Input file in tsv format loaded from data_dir",
    )
parser.add_argument(
    "--translated",
    type=bool,
    default=True,
    help="if a separate input file is provided",
)
args = parser.parse_args()
mnli = MultiNLIPipeline(args=args)
print(args)
mnli()
