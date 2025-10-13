"""
MorphyNet related analys and convertion scripts.
"""

import argparse
import src.morphynet_validation as mval
import src.morphynet_transform as mtran
import src.morphynet_stats as mstats

# default path to original morphynet files
_MORPHYNET_DER_FPATH="./data/morphynet/por.derivational.v1.tsv"
_MORPHYNET_INF_FPATH="./data/morphynet/pt.inflectional.v1.tsv"


def read_arguments():
    '''Read and return valid command line arguments.'''
    parser = argparse.ArgumentParser(description="Process MorphyNet TSV files.")

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
        '-V', '--validate',
        action="store_true",
        default=False,
        help="Run the file validation script.",
    )
    script_options.add_argument(
        '-S', '--stats',
        action="store_true",
        default=False,
        help="Compute file statistics",
    )
    script_options.add_argument(
        '-T', '--transform',
        action="store_true",
        default=False,
        help="Filter and Transform MorphyNet format.",
    )

    # INPUT and OUTPUT options
    parser.add_argument(
        "--file-path",
        required=False,
        help="Path to the TSV file to be processed. When not specified, the" +
            "MorphyNet file will be choosen based on format option."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Folder where generated files will be saved."
    )

    args = parser.parse_args()

    # default input
    if args.file_path is None:
        if args.derivation:
            args.file_path = _MORPHYNET_DER_FPATH
        else:
            args.file_path = _MORPHYNET_INF_FPATH

    return args


def run_derivation_script(args):
    '''Run the script on derivation input format.'''
    input_path = args.file_path
    output_dir = args.output_dir

    if args.validate:
        mval.validate_derivation_file(input_path, output_dir)
    elif args.stats:
        mstats.compute_derivation_stats(input_path, output_dir)
    elif args.transform:
        mtran.transform_derivation_file(input_path, output_dir)
    else:
        print("Invalid Script Option")


def run_inflection_script(args):
    '''Run the script on inflection input format.'''
    input_path = args.file_path
    output_dir = args.output_dir

    if args.validate:
        mval.validate_inflection_file(input_path, output_dir)
    elif args.stats:
        mstats.compute_inflection_stats(input_path, output_dir)
    elif args.transform:
        mtran.transform_inflection_file(input_path, output_dir)
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
