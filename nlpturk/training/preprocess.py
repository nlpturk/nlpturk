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
from ..utils import batch_dataset, split_dataset, lower, isupper, capitalize


def _parse_conllu(filepath: Union[str, Path], min_tokens: int = 5) -> List[Dict[str, List[str]]]:
    """Read conllu file, merge subtokens and normalize lemmas. 
    Lemma cases are inconsistent among treebanks.

    Args:
        filepath (Union[str, Path]): Conllu file path.
        min_tokens (int, optional): Minimum number of tokens the sentence should have.

    Returns:
        List[List[str]]: List of parsed sentences.
    """
    def normalize_lemma(token, lemma):
        i, prefix = 1, ''
        while i <= len(lemma) and lower(token[:i]) == lower(lemma[:i]):
            prefix = token[:i]
            i += 1
        return prefix + lemma[len(prefix):]

    sents = []
    for lines in FS.read(filepath):
        exclude = []
        for idx, line in enumerate(lines):
            if isinstance(line, str):
                exclude.append(idx)
                continue
            # merge subtokens
            if '-' in line[0]:
                lines[idx+1][1] = line[1]
                exclude.extend([idx, idx+2])
            # normalize lemma
            lines[idx][2] = normalize_lemma(line[1], line[2])
        lines = [l for idx, l in enumerate(lines) if idx not in exclude]
        if len(lines) >= min_tokens:
            # some sents are all uppercase, lower tokens and lemmas
            if all(isupper(l[1]) for l in lines):
                for idx, line in enumerate(lines):
                    lines[idx][1] = lower(line[1])
                    lines[idx][2] = lower(line[2])
            sent = {k: [] for k in ('words', 'lemmas', 'poses', 'morphs')}
            for line in lines:
                _, word, lemma, pos, _, morph, *_ = line
                sent['words'].append(word)
                sent['lemmas'].append(lemma)
                sent['poses'].append(pos)
                sent['morphs'].append(morph)
            sents.append(sent)
    return sents


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
    """Extract tags from UD sentences and convert spaCy Doc objects.

    Args:
        sents (List[List[Dict[str, List[str]]]]): List of UD sentence batches.

    Returns:
        Tuple[List[Doc], Dict[str, int]]: List of spaCy Doc objects and data statistics.
    """
    nlp = spacy.blank('tr')
    nlp.tokenizer = Tokenizer(nlp.vocab)

    docs, stats = [], {'sents': 0, 'tokens': 0}
    for sent_group in sents:
        words, lemmas, poses, morphs, sbd_tags = [], [], [], [], []
        for sent in sent_group:
            if random.choice([True, False]):
                # remove EOS punctutation marks
                idx = len(sent['words']) - 1
                while idx and not re.search(r'[^\W_]', sent['words'][idx]):
                    idx -= 1
                sent = {k: v[:idx+1] for k, v in sent.items()}
            if random.choice([True, False]):
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

        if random.choice([True, False]):
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
    """Extract sentence boundaries and convert spaCy Doc objects.

    Args:
        sents (List[List[Dict[str, List[str]]]]): List of sentence batches.

    Returns:
        Tuple[List[Doc], Dict[str, int]]: List of spaCy Doc objects and data statistics.
    """
    nlp = spacy.blank('tr')
    nlp.tokenizer = Tokenizer(nlp.vocab)

    docs, stats = [], {'sents': 0, 'tokens': 0}
    for sent_group in sents:
        words, tags = [], []
        for sent in sent_group:
            tokens = sent['words']
            if random.choice([True, False]):
                # remove EOS punctutation marks
                while tokens and not re.search(r'[^\W_]', tokens[-1]):
                    tokens.pop()
            if not tokens:
                continue
            if random.choice([True, False]):
                # lower first token of sentence
                tokens[0] = lower(tokens[0])

            idx = 1
            while idx < len(tokens) and not re.search(r'[^\W_]', tokens[-idx]):
                idx += 1
            tags.extend(['O']*len(tokens))
            tags[-idx] = 'EOS'
            words.extend(tokens)

            stats['sents'] += 1

        if random.choice([True, False]):
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


def convert(data_path: Union[str, Path], output_path: Union[str, Path]) -> None:
    """Read raw data contents, extract tags and convert to spaCy binary files for model training.

    Args:
        data_path (Union[str, Path]): Raw data files directory.
        output_path (Union[str, Path]): Output path to save binary files. 
    """
    nlp = spacy.blank('tr')
    nlp.tokenizer = Tokenizer(nlp.vocab)
    msg = Printer()

    sents, is_sbd_dataset = [], False
    for filepath in glob.glob(os.path.join(data_path, '**', '*.*'), recursive=True):
        if FS.split_path(filepath)[0] == 'conllu':
            sents.extend(_parse_conllu(filepath))
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
    train, dev, test = split_dataset(sents, (0.8, 0.1, 0.1))

    # convert data to binary and write to disk
    for filename, sents in {'train': train, 'dev': dev, 'test': test}.items():
        docs, stats = _process_sbd(sents) if is_sbd_dataset else _process_ud(sents)
        _doc2bin(docs, os.path.join(output_path, filename + '.spacy'))
        # print stats
        header = capitalize(filename) + ':'
        msg.info(f'{header} {stats["sents"]} sentences, {stats["tokens"]} tokens.')
