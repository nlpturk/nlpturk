import sys
import argparse

from .utils import fetch_ud_treebanks, merge_ud_treebanks
from .training.preprocess import convert


CMD_HELP = '''
Commands:
  fetch_ud  Fetch Universal Dependencies treebanks.   
  merge_ud  Merge Universal Dependencies treebanks.
  convert   Preprocess dataset and convert to binary.
'''


def main():
    parser = argparse.ArgumentParser(
        description='nlpTurk Command-line Interface',
        usage='python -m nlpturk COMMAND [ARGS] [OPTIONS]',
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        'cmd',
        choices=['fetch_ud', 'merge_ud', 'convert'],
        help=CMD_HELP
    )

    # Conditionally required arguments
    parser.add_argument(
        '--data_path',
        type=str,
        required='merge_ud' in sys.argv
                 or 'convert' in sys.argv,
        help='Required for merge_ud, convert commands.'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        required='fetch_ud' in sys.argv
                 or 'merge_ud' in sys.argv
                 or 'convert' in sys.argv,
        help='Required for fetch_ud, merge_ud, convert commands.'
    )

    # Optional arguments
    parser.add_argument(
        '--blacklist',
        type=str,
        required=False,
        help='Comma seperated directory or file names to be excluded.'
    )

    args = parser.parse_args()

    kwargs = {}
    if args.cmd == 'fetch_ud':
        fetch_ud_treebanks(args.output_path)
    elif args.cmd == 'merge_ud':
        if getattr(args, 'blacklist') is not None:
            value = getattr(args, 'blacklist')
            kwargs['blacklist'] = [d.strip() for d in value.split(',')]
        merge_ud_treebanks(args.data_path, args.output_path, **kwargs)
    elif args.cmd == 'convert':
        convert(args.data_path, args.output_path)


if __name__ == '__main__':
    main()
