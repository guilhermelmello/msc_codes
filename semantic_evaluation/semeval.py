"""
SemEval scripts.
"""

import argparse
import sys

from src import semeval


# default path to SemEval files
_SEMEVAL_DER_FPATH="./data/semeval/derivations.tsv"
_SEMEVAL_INF_FPATH="./data/semeval/inflections.tsv"


def read_arguments():
    '''Read and return valid command line arguments.'''
    parser = argparse.ArgumentParser(description="SemEval scripts.")

    # FORMAT OPTIONS
    format_options = parser.add_mutually_exclusive_group(required=True)
    format_options.add_argument(
        '-I', '--inflection',
        action="store_true",
        help="input file contains inflections"
    )
    format_options.add_argument(
        '-D', '--derivation',
        action="store_true",
        help="input file contains derivations"
    )

    # SCRIPT OPTIONS
    script_options = parser.add_mutually_exclusive_group(required=True)
    script_options.add_argument(
        '-A', '--affixes',
        action="store_true",
        default=False,
        help="Extract affixes frequency.",
    )
    script_options.add_argument(
        '-E', '--evaluate',
        action="store_true",
        default=False,
        help="Evaluate a tokenizer on SemEval dataset.",
    )

    # INPUT and OUTPUT options
    parser.add_argument(
        "--file-path",
        required=False,
        help="Path to the TSV file to be processed. When not specified, the" +
            "default SemEval file will be selected based on format option."
    )
    parser.add_argument(
        "--output-dir",
        required=False,
        help="Folder where results will be saved."
    )
    parser.add_argument(
        "--model-name",
        required=False,
        help="The name of the model hosted in HuggingFace's Hub."
    )

    args = parser.parse_args()

    # default input
    if args.file_path is None:
        if args.derivation:
            args.file_path = _SEMEVAL_DER_FPATH
        else:
            args.file_path = _SEMEVAL_INF_FPATH

    return args


def run_derivation_script(args):
    '''Run the script on derivation input format.'''
    input_path = args.file_path
    output_dir = args.output_dir

    if args.affixes:
        if output_dir is None:
            print("The --output-dir parameter is required for --affixes")
            sys.exit(1)
        semeval.affixes_stats.process_derivations(input_path, output_dir)
    elif args.evaluate:
        if args.model_name is None:
            print("The --model-name parameter is required for --evaluate")
            sys.exit(1)
        semeval.evaluate.evaluate(input_path, args.model_name)
    else:
        print("Invalid Script Option")


def run_inflection_script(args):
    '''Run the script on inflection input format.'''
    input_path = args.file_path
    output_dir = args.output_dir

    if args.affixes:
        if output_dir is None:
            print("The --output-dir parameter is required for --affixes")
            sys.exit(1)
        semeval.affixes_stats.process_inflections(input_path, output_dir)
    elif args.evaluate:
        if args.model_name is None:
            print("The --model-name parameter is required for --evaluate")
            sys.exit(1)
        semeval.evaluate.evaluate(input_path, args.model_name)
    else:
        print("Invalid Script Option")


def run():
    '''Script runner entry point.'''
    args = read_arguments()

    if args.derivation:
        run_derivation_script(args)
    elif args.inflection:
        run_inflection_script(args)
    else:
        print("Invalid Format Option")


if __name__ == "__main__":
    run()
