import os
import glob
import shutil
from pathlib import Path
from typing import Union, Dict, List, Any

from spacy.cli.train import train
from spacy.cli.evaluate import evaluate

from ..fs import FS
from ..pipeline import sbd
from .configs import load_default_configs


def _ensure_paths(paths: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure that the paths are valid. 
    """
    if not os.path.isdir(paths['data']):
        raise ValueError(f'Path `{paths["data"]}` does not exist.')
    for k in ('vectors', 'source', 'checkpoint'):
        if k in paths and paths[k] and not os.path.isdir(paths[k]):
            raise ValueError(f'Path `{paths[k]}` does not exist.')
    if not isinstance(paths['model'], (str, Path)):
        raise ValueError(f'`{paths["model"]}` is not a valid path.')

    paths['model-best'] = os.path.join(paths['model'], 'model-best')
    paths['model-last'] = os.path.join(paths['model'], 'model-last')

    paths['tmp'] = os.path.join(paths['model'], 'tmp')
    os.makedirs(paths['tmp'], exist_ok=True)
    paths['config'] = os.path.join(paths['tmp'], 'config.cfg')

    paths.update({'train': None, 'dev': None, 'test': None})
    for filepath in glob.glob(os.path.join(paths['data'], '**', '*.spacy'), recursive=True):
        _, filename, _ = FS.split_path(filepath)
        if 'train' in filename:
            paths['train'] = filepath
        elif 'dev' in filename:
            paths['dev'] = filepath
        elif 'test' in filename:
            paths['test'] = filepath

    if not all((paths['train'], paths['dev'], paths['test'])):
        missing = []
        for k in ('train', 'dev', 'test'):
            if not paths[k]:
                missing.append(k)
        missing = ', '.join(missing)
        raise ValueError(f'Path(s) for `{missing}` do(es) not exist.')

    return paths


def train_model(
    model_path: Union[str, Path],
    data_path: Union[str, Path],
    trf_model: str = None,
    use_gpu: bool = False,
    vectors: Union[str, Path] = None,
    source: Union[str, Path] = None,
    checkpoint: Union[str, Path] = None,
    components: List[str] = None,
    frozen: List[str] = None
) -> None:
    """Train a nlpTurk model. 

    Args:
        model_path (Union[str, Path]): Path to the directory to save trained model. 
            Will be created if it doesn’t exist. 
        data_path (Union[str, Path]): Path to the binary training files.
        trf_model (str, optional): Transformer model name. If specified, pretrained   
            transformer model will be used during training. Defaults to None.
        use_gpu (bool, optional): Whether to use GPU during training. Defaults to False.
        vectors (Union[str, Path], optional): Path to the pretrained word vectors, 
            e.g. fasttext, glove. Word vectors will be used during training if specified.
            Will not be used if transformer model name is supplied. Defaults to None.
        source (Union[str, Path], optional): Path to the existing trained model for 
            incremental training. Pipeline components will be initialized with sourced 
            model weights. Defaults to None.
        checkpoint (Union[str, Path], optional): Path to the checkpoint directory 
            to resume training. Defaults to None.
        components (List[str], optional): Pipeline components to train. If not specified, 
            model will be trained on default pipeline components. This argument is mainly
            used with `source` argument for incremental training. Defaults to None.
        frozen (List[str], optional): Pipeline components to be not updated during
            training. Needs a sourced model. Defaults to None.
    """
    use_gpu = 0 if use_gpu else -1
    configs = load_default_configs(trf_model)
    paths = _ensure_paths({'model': model_path, 'data': data_path, 'vectors': vectors,
                           'source': source, 'checkpoint': checkpoint})

    configs['paths']['train'] = paths['train']
    configs['paths']['dev'] = paths['dev']

    if paths['vectors'] and 'tok2vec' in configs['components']:
        configs['paths']['vectors'] = paths['vectors']
        configs['components']['tok2vec']['model']['embed']['include_static_vectors'] = True
    if paths['source']:
        configs['paths']['source'] = paths['source']
        if 'tok2vec' in configs['components']:
            del configs['components']['tok2vec']['factory']
            configs['components']['tok2vec']['source'] = '${paths.source}'
        elif 'transformer' in configs['components']:
            del configs['components']['transformer']['factory']
            configs['components']['transformer']['source'] = '${paths.source}'

    if components and isinstance(components, list):
        default = {'sbd': 'sbd_f', 'tagger': 'tag_acc', 'lemmatizer': 'lemma_acc'}
        if not all(p in default for p in components):
            raise ValueError('`components` must be consist of ' + ', '.join(default))
        weight = round(1./len(components), 2)
        for c, m in default.items():
            if c not in components:
                configs['nlp']['pipeline'].remove(c)
                del configs['components'][c]
                del configs['training']['score_weights'][m]
            else:
                configs['training']['score_weights'][m] = weight

    if frozen and isinstance(frozen, list) and paths['source']:
        if not paths['source']:
            raise ValueError('A sourced model is need to freeze components.')
        pipes = [k for k in configs['components'] if k not in ('tok2vec', 'transformer')]
        if not all(p in pipes for p in frozen):
            raise ValueError('`frozen` must be consist of ' + ', '.join(pipes))
        update = [p for p in pipes if p not in frozen]
        if not update:
            raise ValueError('All the pipeline components cannot be frozen.')
        replace_listeners = ['model.tok2vec'] if 'tok2vec' in configs['components'] \
            else ['model.transformer']
        weight = round(1./len(update), 2)
        for c, m in {'sbd': 'sbd_f', 'tagger': 'tag_acc', 'lemmatizer': 'lemma_acc'}.items():
            if c in frozen:
                configs['components'][c] = {'source': '${paths.source}',
                                            'replace_listeners': replace_listeners}
                configs["training"]["frozen_components"].append(c)
                configs['training']['score_weights'][m] = 0.0
            else:
                configs['training']['score_weights'][m] = weight

    if paths['checkpoint']:
        for k in configs['components']:
            configs['components'][k] = {"source": paths['checkpoint']}

    # write updated configs to disk
    configs.to_disk(paths['config'])

    # Train the model.
    train(paths['config'], paths['model'], use_gpu=use_gpu)

    # Apply the best and last models to unseen text and measure accuracy.
    evaluate_model(paths['model-best'], paths['test'], os.path.join(paths['model-best'], 'eval.json'),
                   use_gpu=(use_gpu == 0))
    evaluate_model(paths['model-last'], paths['test'], os.path.join(paths['model-last'], 'eval.json'),
                   use_gpu=(use_gpu == 0))

    # remove temporary files
    shutil.rmtree(paths['tmp'], ignore_errors=True)


def evaluate_model(
    model_path: Union[str, Path],
    filepath: Union[str, Path],
    output_path: Union[str, Path] = None,
    use_gpu: bool = False
) -> Dict[str, Any]:
    """Evaluate a trained model.

    Args:
        model_path (Union[str, Path]): Path to the trained model directory.
        filepath (Union[str, Path]): Path to the binary test file.
        output_path (Union[str, Path]): Output path to save evaluation results.
            If not specified, only the scores are returned. Defaults to None.
        use_gpu (bool, optional): Whether to use GPU during evaluation. Defaults to False.

    Returns:
        Dict[str, Any]: Evaluation scores.
    """
    use_gpu = 0 if use_gpu else -1
    if not os.path.isfile(filepath):
        raise ValueError(f'Path `{filepath}` does not exist.')
    scores = evaluate(model_path, filepath, output=output_path, use_gpu=use_gpu)
    return scores
