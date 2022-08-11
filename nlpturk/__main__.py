import argparse
import warnings

from .utils import fetch_ud_treebanks, merge_ud_treebanks
from .training.preprocess import convert
from .training.train import train_model, evaluate_model
from benchmarks.utils import run_benchmarks


_cli_usage = 'Usage: python -m nlpturk [OPTIONS] COMMAND [ARGS]'

_cli_help = _cli_usage + '''

nlpTurk Command-line Interface

Options:
  -h, --help  Show this message and exit.

Commands:
  fetch_ud    Fetch Universal Dependencies treebanks.
            
              Required arguments:
                --output_path  Output path to save fetched UD treebanks.

  merge_ud    Merge Universal Dependencies treebanks.
            
              Required arguments:
                --data_path    Path to the UD treebanks.
                --output_path  Output path to save merged files. 
                               Merged dataset would be splitted into 
                               train, dev, test sets if path is a directory.
            
              Optional arguments:
                --blacklist    Files and directories to be excluded, 
                               e.g. `--blacklist file_name dir_name`. 

  preprocess  Preprocess raw data files and convert to binary.
            
              Required arguments:
                --data_path    Raw data files directory.
                --output_path  Output path to save binary files.
            
              Optional arguments:
                --split_ratios Train, dev, test split ratios in the [0, 1] range. 
                               Either two or one of the ratios can be set to 0.0, 
                               e.g. `--split_ratios 0.8 0.0 0.2`. 

  train       Train a nlpTurk model.
            
              Required arguments:
                --model_path   Path to the directory to save trained model. 
                               Will be created if it doesn’t exist. 
                --data_path    Path to the binary training files.
            
              Optional arguments:
                --trf_model    Transformer model name. If specified, pretrained   
                               transformer model will be used during training.  
                --use_gpu      Flag indicates whether to use GPU during training.
                --vectors      Path to the pretrained word vectors, e.g. fasttext, 
                               glove. Word vectors will be used during training.
                --source       Path to the existing trained model for incremental 
                               training. Pipeline components will be initialized 
                               with sourced model weights.   
                --checkpoint   Path to the checkpoint directory to resume training. 
                --components   Pipeline components to train. If not specified, 
                               model will be trained on default pipeline components. 
                               This argument is mainly used with `--source` argument 
                               for incremental training.
                --frozen       Pipeline components to be not updated during training. 
                               Needs a sourced model.   

  evaluate    Evaluate a trained model.
            
              Required arguments:
                --model_path   Path to the trained model directory. 
                --filepath     Path to the binary test file.
                --output_path  Output path to save evaluation results.
            
              Optional arguments:  
                --use_gpu      Flag indicates whether to use GPU during evaluation.

  benchmark   Perform benchmarks.
            
              Required arguments:
                --data_path    Path to the file or directory of files. Multiple files 
                               will be merged. If files are in conllu format, benchmarks  
                               will be performed for sentence segmentation, lemmatization 
                               and POS tagging. If files contains sentences seperated by 
                               newlines, benchmarks will be performed only for sentence 
                               segmentation.
                --output_path  Output path to save benchmark report.

Usage Examples: 
  python -m nlpturk preprocess --data_path path/to/data --output_path path/to/output 
  python -m nlpturk train --model_path path/to/model --data_path path/to/data --use_gpu
'''


class ArgumentParser(argparse.ArgumentParser):

    def __init__(self, *args, **kwargs):
        super(ArgumentParser, self).__init__(*args, **kwargs)

    def parse_args(self, args=None, namespace=None):
        args, argv = self.parse_known_args(args, namespace)
        for i, arg in enumerate(argv):
            if arg.startswith('--'):
                arg = arg.lstrip('--')
                if '=' in arg:
                    k, v = arg.split('=', 1)
                    setattr(args, k, v)
                else:
                    val = []
                    while i+1 < len(argv) and not argv[i+1].startswith('--'):
                        val.append(argv[i+1])
                        i += 1
                    val = '' if not val else val if len(val) > 1 else val[0]
                    setattr(args, arg, val)

        return args

    def print_usage(self):
        print(_cli_usage)

    def print_help(self):
        print(_cli_help)

    def error(self, message):
        self.print_usage()
        self.exit(2, f'nlpturk: error: {message}\n')


def cli():
    E01 = 'the following arguments are required: {}'
    CMD = ['fetch_ud', 'merge_ud', 'preprocess', 'train', 'evaluate', 'benchmark']

    parser = ArgumentParser()
    parser.add_argument('COMMAND', choices=CMD)
    args = parser.parse_args()

    warnings.filterwarnings('ignore')

    kwargs = {}
    if args.COMMAND == 'fetch_ud':
        if not hasattr(args, 'output_path') or not args.output_path:
            parser.error(E01.format('--output_path'))
        fetch_ud_treebanks(args.output_path)
    elif args.COMMAND == 'merge_ud':
        required = [a for a in ('data_path', 'output_path')
                    if not hasattr(args, a) or not getattr(args, a)]
        if required:
            parser.error(E01.format(', '.join([f'--{r}' for r in required])))
        if hasattr(args, 'blacklist') and args.blacklist:
            kwargs['blacklist'] = args.blacklist if isinstance(args.blacklist, list) \
                else [args.blacklist]
        merge_ud_treebanks(args.data_path, args.output_path, **kwargs)
    elif args.COMMAND == 'preprocess':
        required = [a for a in ('data_path', 'output_path')
                    if not hasattr(args, a) or not getattr(args, a)]
        if required:
            parser.error(E01.format(', '.join([f'--{r}' for r in required])))
        if hasattr(args, 'split_ratios') and isinstance(args.split_ratios, list):
            kwargs['split_ratios'] = [float(r) for r in args.split_ratios]
        convert(args.data_path, args.output_path)
    elif args.COMMAND == 'train':
        required = [a for a in ('model_path', 'data_path')
                    if not hasattr(args, a) or not getattr(args, a)]
        if required:
            parser.error(E01.format(', '.join([f'--{r}' for r in required])))
        if hasattr(args, 'use_gpu'):
            kwargs['use_gpu'] = True
        if hasattr(args, 'components') and args.components:
            kwargs['components'] = args.components if isinstance(args.components, list) \
                else [args.components]
        if hasattr(args, 'frozen') and args.frozen:
            kwargs['frozen'] = args.frozen if isinstance(args.frozen, list) \
                else [args.frozen]
        for name in ['trf_model', 'vectors', 'source', 'checkpoint']:
            if hasattr(args, name) and getattr(args, name):
                kwargs[name] = getattr(args, name)
        train_model(args.model_path, args.data_path, **kwargs)
    elif args.COMMAND == 'evaluate':
        required = [a for a in ('model_path', 'filepath', 'output_path')
                    if not hasattr(args, a) or not getattr(args, a)]
        if required:
            parser.error(E01.format(', '.join([f'--{r}' for r in required])))
        if hasattr(args, 'use_gpu'):
            kwargs['use_gpu'] = True
        evaluate_model(args.model_path, args.filepath, args.output_path, **kwargs)
    elif args.COMMAND == 'benchmark':
        required = [a for a in ('data_path', 'output_path')
                    if not hasattr(args, a) or not getattr(args, a)]
        if required:
            parser.error(E01.format(', '.join([f'--{r}' for r in required])))
        run_benchmarks(args.data_path, args.output_path)


if __name__ == '__main__':
    cli()
