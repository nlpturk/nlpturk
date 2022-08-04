import os
import glob
import shutil
import requests
from pathlib import Path
from zipfile import ZipFile
from typing import Iterator, Tuple, Iterable, Union, List, Any

from sklearn.model_selection import train_test_split
from wasabi import Printer

from .fs import FS


def lower(text: str) -> str:
    """Convert string to lowercase, fixes Turkish "İ" and "ı" problems.
    """
    return text.replace('I', 'ı').replace('İ', 'i').lower()


def upper(text: str) -> str:
    """Convert string to uppercase, fixes Turkish "İ" and "ı" problems.
    """
    return text.replace('i', 'İ').upper()


def capitalize(text: str) -> str:
    """Capitalize first character in string, fixes Turkish "İ" and "ı" problems.
    """
    text = lower(text)
    return text.replace(text[0], upper(text[0]), 1)


def capwords(text: str) -> str:
    """Capitalize each word in string, fixes Turkish "İ" and "ı" problems.
    """
    return ' '.join(capitalize(word) for word in text.split())


def islower(text: str) -> str:
    """Check if string contains all lowercase characters 
    """
    return text == lower(text)


def isupper(text: str) -> str:
    """Check if string contains all uppercase characters 
    """
    return text == upper(text)


def istitle(text: str) -> str:
    """Check if each word in string starts with an uppercase letter 
    """
    return all(word == capitalize(word) for word in text.split())


def batch_dataset(data: Iterable[Any], batch_size: int = 1) -> Iterator[Iterable[Any]]:
    """Split dataset into batches. Last batch would not be padded. 

    Args:
        data (Iterable[Any]): Data to be splitted.
        batch_size (int, optional): Batch size. Defaults to 1.

    Yields:
        Iterator[Iterable[Any]]: Yield batches, one for each batch.
    """
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('"batch_size" must be a positive integer.')
    l = len(data)
    for ndx in range(0, l, batch_size):
        yield data[ndx:min(ndx + batch_size, l)]


def split_dataset(
    data: List[Any],
    split_ratios: Union[List[float], Tuple[float, ...]]
) -> Tuple[Union[List[Any], None], Union[List[Any], None], Union[List[Any], None]]:
    """Split dataset into train, dev, test sets.

    Args:
        data (List[Any]): Dataset to be splitted.
        split_ratios (Union[List[float], Tuple[float, ...]]): Train, dev, test split ratios 
            in the [0, 1] range. Either two or one of the ratios can be set to 0.0.

    Returns:
        Tuple[Union[List[Any], None], Union[List[Any], None], Union[List[Any], None]]: 
            Splitted train, dev, test sets.
    """
    if not data or not isinstance(data, list):
        raise ValueError('Data must be list of values.')

    if not isinstance(split_ratios, (tuple, list)) or len(split_ratios) != 3 or \
            not all([isinstance(v, float) and v >= 0. for v in split_ratios]) or \
            sum(split_ratios) != 1.0:
        raise ValueError(
            'Train, dev, test split ratios must be float in the [0, 1] range and the sum must be 1.0.')

    splits = {k: split_ratios[i] for i, k in enumerate(['train', 'dev', 'test'])}
    nr_splits = len([v for v in splits.values() if v])

    if nr_splits == 3:
        splits['dev'] = splits['dev']/(1 - splits['test'])
        train, test = train_test_split(data, test_size=splits['test'], random_state=42)
        train, dev = train_test_split(train, test_size=splits['dev'], random_state=42)
        return train, dev, test
    elif nr_splits == 2:
        if not splits['train']:
            dev, test = train_test_split(data, test_size=splits['test'],
                                         random_state=42)
            return None, dev, test
        elif not splits['dev']:
            train, test = train_test_split(data, test_size=splits['test'],
                                           random_state=42)
            return train, None, test
        else:
            train, dev = train_test_split(data, test_size=splits['dev'],
                                          random_state=42)
            return train, dev, None
    else:
        return (
            data if splits['train'] else None,
            data if splits['dev'] else None,
            data if splits['test'] else None
        )


def fetch_ud_treebanks(output_path: Union[str, Path]) -> None:
    """Fetch Universal Dependencies treebanks.

    Args:
        output_path (Union[str, Path]): Output path to save fetched UD treebanks.
    """
    base_url = 'https://github.com/UniversalDependencies/'
    repos = {
        'atis': 'UD_Turkish-Atis',
        'boun': 'UD_Turkish-BOUN',
        'framenet': 'UD_Turkish-FrameNet',
        'imst': 'UD_Turkish-IMST',
        'kenet': 'UD_Turkish-Kenet',
        'penn': 'UD_Turkish-Penn',
        'tourism': 'UD_Turkish-Tourism'
    }

    for k, v in repos.items():
        url = base_url + v + '/archive/refs/heads/master.zip'

        repo_dir = os.path.join(output_path, k)
        os.makedirs(repo_dir, exist_ok=True)

        # create temporary directory to fetch all the files in repository
        tmp_dir = os.path.join(output_path, 'ud_tmp')
        os.makedirs(tmp_dir, exist_ok=True)
        zip_file = os.path.join(tmp_dir, 'ud.zip')

        try:
            res = requests.get(url, allow_redirects=True)
            with open(zip_file, 'wb') as f:
                f.write(res.content)
        except Exception as e:
            with open(os.path.join(output_path, 'error.log'), 'a+') as f:
                f.write(f'url: {url}\nmsg: {e}\n\n')

        with ZipFile(zip_file, 'r') as f:
            f.extractall(tmp_dir)

        for filepath in glob.glob(os.path.join(tmp_dir, '**', '*.conllu'), recursive=True):
            shutil.move(filepath, repo_dir)

        # remove temporary directory
        shutil.rmtree(tmp_dir, ignore_errors=True)


def merge_ud_treebanks(
    data_path: Union[str, Path],
    output_path: Union[str, Path],
    blacklist: Iterable[str] = None
) -> None:
    """Merge Universal Dependencies treebanks.

    Args:
        data_path (Union[str, Path]): Path to the UD treebanks.
        output_path (Union[str, Path]): Output path to save merged files. Merged dataset 
            would be splitted into train, dev, test sets if path is a directory.
        blacklist (Iterable[str], optional): Files and directories to be excluded. Defaults to None.
    """
    data = []
    for filepath in glob.glob(os.path.join(data_path, '**', '*.conllu'), recursive=True):
        if isinstance(blacklist, (list, tuple, set)) and \
                not all(n not in filepath.lstrip(str(data_path)) for n in blacklist):
            continue
        for lines in FS.read(filepath):
            ext, filename, _ = FS.split_path(filepath)
            # append source of each sentence as comment
            sent = [f'# source = {filename}.{ext}']
            for line in lines:
                line = line if isinstance(line, str) else '\t'.join(line)
                sent.append(line)
            data.append('\n'.join(sent))

    msg = Printer()
    msg.info(f'# of sentences: {len(data)}')

    if FS.split_path(output_path)[0]:
        # output is file
        FS.to_disk('\n\n'.join(data), output_path)
    else:
        # output is directory
        # split into train, dev, test sets
        train, dev, test = split_dataset(data, (0.8, 0.1, 0.1))
        FS.to_disk('\n\n'.join(train), os.path.join(output_path, 'train.conllu'))
        FS.to_disk('\n\n'.join(dev), os.path.join(output_path, 'dev.conllu'))
        FS.to_disk('\n\n'.join(test), os.path.join(output_path, 'test.conllu'))

        msg.info(f'# of train sentences: {len(train)}')
        msg.info(f'# of validation sentences: {len(dev)}')
        msg.info(f'# of test sentences: {len(test)}')
