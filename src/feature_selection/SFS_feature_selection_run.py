import sys
import argparse

sys.path.append('src')
from feature_selection.SFSSelector import SFS_Selector

### Menue ###
parser = argparse.ArgumentParser(description='Sequential Feature Selection for the best 20 Features.',
                                 add_help=True)
parser.add_argument('-i', '--input', help='Path to csv file containing features of interest.')
parser.add_argument('-p', '--predictive_value', help='Name of the predictive value as the label.')
parser.add_argument('-l', '--label_file', default="",
                    help='If label not in file with radiomics features then provide path to CSV file with labels.')
parser.add_argument('-c', '--cpu', default=1, type=int, help='Number of CPUs to use.')
parser.add_argument('-o', '--output', help='Path to folder for output.')

args = vars(parser.parse_args())

INPUT = args["input"]
PREDICTIVE_VALUE = args["predictive_value"]
LABEL_FILE = args["label_file"]
CPU = args["cpu"]
OUT = args["output"]

selector = SFS_Selector(input=INPUT,
                        predictive_value=PREDICTIVE_VALUE,
                        label_file=LABEL_FILE,
                        cpu=CPU,
                        out_path=OUT)

selector.run()
