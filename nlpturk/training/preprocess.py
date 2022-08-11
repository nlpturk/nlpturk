import re
import os
import glob
import random
from pathlib import Path
from typing import Union, Tuple, List, Dict

import spacy
from spacy.tokens import Doc, DocBin
from wasabi import Printer

from ..fs import FS
from ..pipeline.tokenizer import Tokenizer
from ..utils import batch_dataset, split_dataset, lower, capitalize


def _doc2bin(docs: List[Doc], filepath: Union[str, Path]) -> None:
    """Convert data to spaCy binary and write to disk.

    Args:
        docs (List[Doc]): List of spacy Doc objects.
        filepath (Union[str,Path]): Binary file path.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    docbin = DocBin(docs=docs, store_user_data=True)
    docbin.to_disk(filepath)


def _process_ud(sents: List[List[Dict[str, List[str]]]]) -> Tuple[List[Doc], Dict[str, int]]:
    """Extract tags from UD sentences and convert to spaCy Doc objects.

    Args:
        sents (List[List[Dict[str, List[str]]]]): List of UD sentence batches.

    Returns:
        Tuple[List[Doc], Dict[str, int]]: List of spaCy Doc objects and data statistics.
    """
    nlp = spacy.blank('tr')
    nlp.tokenizer = Tokenizer(nlp)

    docs, stats = [], {'sents': 0, 'tokens': 0}
    for sent_group in sents:
        words, lemmas, poses, morphs, sbd_tags = [], [], [], [], []
        for sent in sent_group:
            if random.choice((0, 1)):
                # remove EOS punctutation marks
                idx = len(sent['words']) - 1
                while idx and not re.search(r'[^\W_]', sent['words'][idx]):
                    idx -= 1
                sent = {k: v[:idx+1] for k, v in sent.items()}
            if random.choice((0, 1)):
                # lower first token of sentence
                sent['words'][0] = lower(sent['words'][0])
                sent['lemmas'][0] = lower(sent['lemmas'][0])

            words.extend(sent['words'])
            lemmas.extend(sent['lemmas'])
            poses.extend(sent['poses'])
            morphs.extend(sent['morphs'])

            idx, l = 1, len(sent['words'])
            while idx < l and not re.search(r'[^\W_]', words[-idx]):
                idx += 1
            sbd_tags.extend(['O']*l)
            sbd_tags[-idx] = 'EOS'

            stats['sents'] += 1

        if random.choice((0, 1)):
            # remove last EOS token and trailing punctuation marks
            idx = 1
            while idx < len(words) and sbd_tags[-idx] != 'EOS':
                idx += 1
            del words[-idx:]
            del lemmas[-idx:]
            del poses[-idx:]
            del morphs[-idx:]
            del sbd_tags[-idx:]

        spaces = [True] * len(words)
        spaces[-1] = False
        doc = Doc(nlp.vocab, words=words, spaces=spaces, lemmas=lemmas,
                  tags=poses, pos=poses, morphs=morphs)
        for t, r in zip(doc, Doc(nlp.vocab, words=words, spaces=spaces, tags=sbd_tags)):
            t._.sbd_tag_ = r.tag_
            t._.sbd_tag = r.tag
        docs.append(doc)

        stats['tokens'] += len(words)

    return docs, stats


def _process_sbd(sents: List[List[Dict[str, List[str]]]]) -> Tuple[List[Doc], Dict[str, int]]:
    """Extract sentence boundaries and convert to spaCy Doc objects.

    Args:
        sents (List[List[Dict[str, List[str]]]]): List of sentence batches.

    Returns:
        Tuple[List[Doc], Dict[str, int]]: List of spaCy Doc objects and data statistics.
    """
    nlp = spacy.blank('tr')
    nlp.tokenizer = Tokenizer(nlp)

    docs, stats = [], {'sents': 0, 'tokens': 0}
    for sent_group in sents:
        words, tags = [], []
        for sent in sent_group:
            tokens = sent['words']
            if random.choice((0, 1)):
                # remove EOS punctutation marks
                while tokens and not re.search(r'[^\W_]', tokens[-1]):
                    tokens.pop()
            if not tokens:
                continue
            if random.choice((0, 1)):
                # lower first token of sentence
                tokens[0] = lower(tokens[0])

            idx = 1
            while idx < len(tokens) and not re.search(r'[^\W_]', tokens[-idx]):
                idx += 1
            tags.extend(['O']*len(tokens))
            tags[-idx] = 'EOS'
            words.extend(tokens)

            stats['sents'] += 1

        if random.choice((0, 1)):
            # remove last EOS token and trailing tokens
            idx = 1
            while idx < len(words) and tags[-idx] != 'EOS':
                idx += 1
            del words[-idx:]
            del tags[-idx:]

        spaces = [True] * len(words)
        spaces[-1] = False
        doc = Doc(nlp.vocab, words=words, spaces=spaces)
        for t, r in zip(doc, Doc(nlp.vocab, words=words, spaces=spaces, tags=tags)):
            t._.sbd_tag_ = r.tag_
            t._.sbd_tag = r.tag
        docs.append(doc)

        stats['tokens'] += len(words)

    return docs, stats


def convert(
    data_path: Union[str, Path],
    output_path: Union[str, Path],
    split_ratios: Union[List[float], Tuple[float, ...]] = (0.8, 0.1, 0.1)
) -> None:
    """Read raw data contents, extract tags and convert to spaCy binary files for model training.

    Args:
        data_path (Union[str, Path]): Raw data files directory.
        output_path (Union[str, Path]): Output path to save binary files.
        split_ratios (Union[List[float], Tuple[float, ...]]): Train, dev, test split ratios 
            in the [0, 1] range. Either or both of the dev and test split ratios can be set to 0.0. 
    """
    nlp = spacy.blank('tr')
    nlp.tokenizer = Tokenizer(nlp)
    msg = Printer()

    sents, is_sbd_dataset = [], False
    for filepath in glob.glob(os.path.join(data_path, '**', '*.*'), recursive=True):
        if FS.split_path(filepath)[0] == 'conllu':
            sents.extend(FS.parse_conllu(filepath))
        else:
            is_sbd_dataset = True
            # sentences are seperated by new lines
            # tokenize sentences, discard whitespace tokens
            sents.extend(
                {'words': [t.text for t in nlp(s) if t.text.strip()]}
                for s in FS.read(filepath).split('\n') if len(s.split()) >= 5
            )

    # group data by 10 sentences
    sents = list(batch_dataset(sents, batch_size=10))
    # split data into train, dev, test sets.
    train, dev, test = split_dataset(sents, split_ratios=split_ratios)

    # convert data to binary and write to disk
    for filename, sents in {'train': train, 'dev': dev, 'test': test}.items():
        if sents:
            docs, stats = _process_sbd(sents) if is_sbd_dataset else _process_ud(sents)
            _doc2bin(docs, os.path.join(output_path, filename + '.spacy'))
            # print stats
            header = capitalize(filename) + ':'
            msg.info(f'{header} {stats["sents"]} sentences, {stats["tokens"]} tokens.')
