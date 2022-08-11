import os
from pathlib import Path
from datetime import datetime
from typing import Union, List, Dict, Any

import nltk
import spacy
from spacy.training import Example
from spacy.scorer import Scorer
from wasabi import Printer
from sklearn.metrics import accuracy_score

import nlpturk
from nlpturk.fs import FS
from nlpturk.pipeline.tokenizer import Tokenizer
from nlpturk.utils import batch_dataset, lower
from .zemberek.base import Zemberek


def run_benchmarks(filepath: Union[str, Path], output_path: Union[str, Path]) -> None:
    """Perform benchmarks.

    Args:
        filepath (Union[str, Path]): Path to the benchmark file. If file is in conllu format,
            benchmarks will be performed for sentence segmentation, lemmatization and POS tagging. 
            If file contains sentences seperated by newlines, benchmarks will be performed  
            only for sentence segmentation.
        output_path (Union[str, Path]): Output path to save benchmark report.
    """
    if not os.path.isfile(filepath):
        raise ValueError(f'Path `{filepath}` does not exist.')

    msg = Printer()

    ext, filename, _ = FS.split_path(filepath)
    filename += f'.{ext}'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if ext == 'conllu':
        sents = FS.parse_conllu(filepath)
    else:
        nlp = spacy.blank('tr')
        nlp.tokenizer = Tokenizer(nlp)
        # sentences are seperated by new lines, tokenize sentences
        sents = [{'words': [t.text for t in nlp(s) if t.text.strip()]}
                 for s in FS.read(filepath).split('\n')]
        sents = [s for s in sents if s['words']]

    if not sents or len(sents) == 1:
        raise ValueError(f'`{filename}` must be comprised of at least two sentences.')

    # group data by 10 sentences
    sents = list(batch_dataset(sents, batch_size=10))

    gold, stats = [], {'sents': 0, 'tokens': 0}
    for group in sents:
        data = {'tokens': [], 'sbd': [], 'lemma': [], 'pos': []}
        for sent in group:
            stats['sents'] += 1
            stats['tokens'] += len(sent['words'])
            data['tokens'].extend(sent['words'])
            data['sbd'].extend(['O']*len(sent['words']))
            # for last token of each sentence label `sbd` as `EOS`
            data['sbd'][-1] = 'EOS'
            if 'lemmas' in sent:
                data['lemma'].extend(lower(l) for l in sent['lemmas'])
            if 'poses' in sent:
                data['pos'].extend(sent['poses'])
        # for last token of each sentence group label `sbd` as `O`
        data['sbd'][-1] = 'O'
        gold.append(data)

    msg.info(f'Parsing file `{filename}` ...')
    msg.info(f'{stats["sents"]} sentences and {stats["tokens"]} tokens found.')

    pred = _predict(gold)

    msg.info('Calculating scores ...')

    scores = {'nltk': {}, 'zemberek': {}, 'nlpTurk': {}}
    for m, p in pred.items():
        # sbd scores
        scores[m]['sbd'] = _score(gold, p, 'sbd')['sbd_per_type']['EOS']
        # lemma scores
        score = _score(gold, p, 'lemma')
        scores[m]['lemma'] = score['lemma_acc'] if score else None
        # pos scores
        score = _score(gold, p, 'pos')
        if score:
            gold_labels = set(t for g in gold for t in g['pos'])
            avg = {k.replace('pos_', ''): v for k, v in score.items()
                   if k != 'pos_per_type'}
            score = {k: v for k, v in score['pos_per_type'].items() if k in gold_labels}
            score.update(avg)
        scores[m]['pos'] = score

    FS.to_disk(_create_report(scores, filename, stats), output_path)

    msg.info(f'Benchmark report saved to `{Path(output_path).resolve()}`')


def _predict(gold: List[Dict[str, List[str]]]) -> Dict[str, List[Dict[str, List[str]]]]:
    """Perform predictions on sentence groups.

    Args:
        gold (List[Dict[str, List[str]]]): List of sentence groups.

    Returns:
        Dict[str, List[Dict[str, List[str]]]]: Predictions.
    """
    msg = Printer()
    predictions = {}

    # nlptTurk predictions
    msg.info('Executing nlpTurk predictions ...')
    pred = []
    for sents in gold:
        data = {'tokens': [], 'sbd': [], 'lemma': [], 'pos': []}
        for token in nlpturk(' '.join(sents['tokens'])):
            data['tokens'].append(token.text)
            data['sbd'].append('EOS' if token.is_sent_end else 'O')
            data['lemma'].append(token.lemma)
            data['pos'].append(token.pos)
        # for last token of each sentence group label `sbd` as `O`
        data['sbd'][-1] = 'O'
        pred.append(data)
    predictions['nlpTurk'] = pred

    # zemberek predictions
    msg.info('Executing zemberek predictions ...')
    zemberek = Zemberek()
    pred = []
    for sents in gold:
        data = {'tokens': [], 'sbd': [], 'lemma': [], 'pos': []}
        for sent in zemberek.extract_sents(' '.join(sents['tokens'])):
            for token in zemberek.extract_morphs(sent):
                data['tokens'].append(token['token'])
                data['sbd'].append('O')
                data['lemma'].append(token['lemma'])
                data['pos'].append(token['pos'])
            data['sbd'][-1] = 'EOS'
        # for last token of each sentence group label `sbd` as `O`
        data['sbd'][-1] = 'O'
        pred.append(data)
    predictions['zemberek'] = pred

    # nltk predictions
    msg.info('Executing NLTK predictions ...')
    predictions['nltk'] = _nltk_tokenize(gold)

    return predictions


def _nltk_tokenize(gold: List[Dict[str, List[str]]]) -> List[Dict[str, List[str]]]:
    """Sentence tokenizer using NLTK's `PunktSentenceTokenizer`.

    Args:
        gold (List[Dict[str, List[str]]]): List of texts to split into sentences.

    Returns:
        List[List[Dict[str, List[str]]]]: Splitted sentences.
    """
    try:
        tokenizer = nltk.data.load('tokenizers/punkt/turkish.pickle')
    except:
        nltk.download('punkt')
        tokenizer = nltk.data.load('tokenizers/punkt/turkish.pickle')

    pred = []
    for sents in gold:
        data = {'tokens': [], 'sbd': [], 'lemma': [], 'pos': []}
        for sent in tokenizer.tokenize(' '.join(sents['tokens'])):
            tokens = sent.split()
            data['tokens'].extend(tokens)
            data['sbd'].extend(['O']*len(tokens))
            data['sbd'][-1] = 'EOS'
        # for last token of each sentence group label `sbd` as `O`
        data['sbd'][-1] = 'O'
        pred.append(data)

    return pred


def _score(
    gold: List[Dict[str, List[str]]],
    pred: List[Dict[str, List[str]]],
    attr: str
) -> Dict[str, Any]:
    """Calculate PRF and accuracy scores.

    Args:
        gold (List[Dict[str, List[str]]]): Gold labels.
        pred (List[Dict[str, List[str]]]): Predicted labels.
        attr (str): The attribute to score. Will be used to prefix score names.

    Returns:
        Dict[str, Any]: PRF and accuracy scores.
    """
    def make_doc(content):
        nlp = spacy.blank('tr')
        nlp.tokenizer = Tokenizer(nlp)

        tokens, spans, idx = [], [], 0
        for token, label in content:
            tokens.append(token)
            spans.append((idx, idx + len(token), label))
            idx += len(token) + 1

        doc = nlp.make_doc(' '.join(tokens))
        doc.spans[attr] = [doc.char_span(s[0], s[1], label=s[2]) for s in spans]

        return doc

    def span_getter(doc, span_key):
        return doc.spans[span_key]

    # if `gold` or `pred` does not have labels, just return
    if not gold[0][attr] or not pred[0][attr]:
        return

    gold = [[(t, l) for t, l in zip(s['tokens'], s[attr])] for s in gold]
    pred = [[(t, l) for t, l in zip(s['tokens'], s[attr])] for s in pred]

    if attr == 'lemma':
        # align tokens
        pred_aligned = []
        for g, p in zip(gold, pred):
            for i, t in enumerate(g):
                if t[0] != p[i][0]:
                    sfx = t[0].lstrip(p[i][0])
                    while sfx:
                        del p[i]
                        sfx = sfx.lstrip(p[i][0])
            pred_aligned.append(p[:len(g)])

        gold = [t[1] for s in gold for t in s]
        pred = [t[1] for s in pred_aligned for t in s]
        scores = {'lemma_acc': accuracy_score(gold, pred)}
    else:
        eg = [Example(make_doc(p), make_doc(g)) for p, g in zip(pred, gold)]
        scores = Scorer.score_spans(eg, attr=attr, getter=span_getter)

    return scores


def _create_report(scores: Dict[str, Any], filename: str, stats: Dict[str, int]) -> str:
    """Creates benchmark report.

    Args:
        scores (Dict[str, Any]): PRF and accuracy scores per library.
        filename (str): Benchmark file name.
        stats (Dict[str, int]): File statistics (# of sents, # of tokens).

    Returns:
        str: Pretty formatted benchmark report.
    """
    report = [f"{'-'*60}\nBENCHMARK REPORT\n{'-'*60}\n"]
    report.append(f'Repository: https://github.com/nlpturk\n')
    report.append(f'Date:  {datetime.today().strftime("%d/%m/%Y")}')
    report.append(f'File:  {filename}')
    report.append(f'Stats: {stats["sents"]} sentences, {stats["tokens"]} tokens')

    # sentence segmentation
    report.append(f"\n\nSentence Segmentation\n{'-'*21}\n")
    report.append(' '*15 + 'precision    recall  f1-score\n')
    for m, p in scores.items():
        sbd = {k: '%0.2f' % (v*100) for k, v in p['sbd'].items()}
        sbd = {k: f"{' '*(10-len(v))}{v}" for k, v in sbd.items()}
        sbd = f"{sbd['p']}{sbd['r']}{sbd['f']}"
        report.append(f"{' '*4}{m}{' '*(10-len(m))}{sbd}")

    # lemmatization
    if any(p['lemma'] is not None for p in scores.values()):
        report.append(f"\n\nLemmatization\n{'-'*13}\n")
        report.append(' '*16 + 'accuracy\n')
        for m, p in scores.items():
            if p['lemma'] is not None:
                acc = '%0.2f' % (p['lemma']*100)
                report.append(f"{' '*4}{m}{' '*(10-len(m))}{' '*(10-len(acc))}{acc}")

    # POS tagging
    pos_tags = {
        'ADJ': 'adjective', 'ADP': 'adposition', 'ADV': 'adverb', 'AUX': 'auxiliary',
        'CCONJ': 'coordinating conjunction', 'DET': 'determiner', 'INTJ': 'interjection',
        'NOUN': 'noun', 'NUM': 'numeral', 'PART': 'particle', 'PRON': 'pronoun',
        'PROPN': 'proper noun', 'PUNCT': 'punctuation', 'SCONJ': 'subordinating conjunction',
        'SYM': 'symbol', 'VERB': 'verb', 'X': 'other'
    }
    if any(p['pos'] is not None for p in scores.values()):
        pos_tags = {k: v for k, v in pos_tags.items() if k in scores['nlpTurk']['pos']}
        report.append(f"\n\nPOS Tagging\n{'-'*11}\n")
        report.append(f"  Micro Average\n  {'-'*13}\n")
        report.append(' '*15 + 'precision    recall  f1-score\n')
        for m, p in scores.items():
            if p['pos'] is not None:
                pos = {k: '%0.2f' % (v*100) for k, v in p['pos'].items() if k in 'prf'}
                pos = {k: f"{' '*(10-len(v))}{v}" for k, v in pos.items()}
                pos = f"{pos['p']}{pos['r']}{pos['f']}"
                report.append(f"{' '*4}{m}{' '*(10-len(m))}{pos}")
        for tag, lf in pos_tags.items():
            lf = f'{tag} [{lf}]'
            report.append(f"\n\n  {lf}\n  {'-'*len(lf)}\n")
            report.append(' '*15 + 'precision    recall  f1-score\n')
            for m, p in scores.items():
                if p['pos'] is not None:
                    pos = {k: '%0.2f' % (v*100) for k, v in p['pos'][tag].items()}
                    pos = {k: f"{' '*(10-len(v))}{v}" for k, v in pos.items()}
                    pos = f"{pos['p']}{pos['r']}{pos['f']}"
                    report.append(f"{' '*4}{m}{' '*(10-len(m))}{pos}")

    return '\n'.join(report)
