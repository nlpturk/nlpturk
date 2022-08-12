import os
import glob
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


def run_benchmarks(data_path: Union[str, Path], output_path: Union[str, Path]) -> None:
    """Perform benchmarks.

    Args:
        data_path (Union[str, Path]): Path to the file or directory of files. Multiple files 
            will be merged. If files are in conllu format, benchmarks will be performed for 
            sentence segmentation, lemmatization and POS tagging. If files contains sentences 
            seperated by newlines, benchmarks will be performed only for sentence segmentation.
        output_path (Union[str, Path]): Output path to save benchmark report.
    """
    if os.path.isfile(data_path):
        files = [data_path]
    elif os.path.isdir(data_path):
        files = glob.glob(os.path.join(data_path, '**', '*.*'), recursive=True)
    else:
        raise ValueError(f'Path `{data_path}` does not exist.')

    nlp = spacy.blank('tr')
    nlp.tokenizer = Tokenizer(nlp)

    # group files by type and merge
    sents = {'conllu': [], 'sbd': []}
    for filepath in files:
        if FS.split_path(filepath)[0] == 'conllu':
            data = FS.parse_conllu(filepath)
            sents['conllu'].extend(data)
            sents['sbd'].extend([{'words': s['words']} for s in data])
        else:
            # sentences are seperated by newlines, tokenize sentences
            data = [{'words': [t.text for t in nlp(s) if t.text.strip()]}
                    for s in FS.read(filepath).split('\n')]
            sents['sbd'].extend([s for s in sents if s['words']])

    if len(sents['sbd']) < 2:
        raise ValueError('Files must be comprised of at least two sentences.')

    filenames = [FS.split_path(f)[1] for f in files]
    stats = {
        'sents': len(sents['sbd']),
        'tokens': sum(len(s['words']) for s in sents['sbd'])
    }

    msg = Printer()
    msg.info(f'Parsing file(s) `{", ".join(filenames)}` ...')
    msg.info(f'{stats["sents"]} sentences and {stats["tokens"]} tokens found.')

    # group data by 10 sentences
    sents['conllu'] = list(batch_dataset(sents['conllu'], batch_size=10))
    sents['sbd'] = list(batch_dataset(sents['sbd'], batch_size=10))

    gold = {'conllu': [], 'sbd': []}
    for group in sents['conllu']:
        data = {'tokens': [], 'lemma': [], 'pos': []}
        for sent in group:
            data['tokens'].extend(sent['words'])
            data['lemma'].extend(lower(l) for l in sent['lemmas'])
            data['pos'].extend(sent['poses'])
        gold['conllu'].append(data)
    for group in sents['sbd']:
        data = {'tokens': [], 'sbd': []}
        for sent in group:
            data['tokens'].extend(sent['words'])
            data['sbd'].extend(['O']*len(sent['words']))
            # for last token of each sentence label `sbd` as `EOS`
            data['sbd'][-1] = 'EOS'
        # for last token of each sentence group label `sbd` as `O`
        data['sbd'][-1] = 'O'
        gold['sbd'].append(data)

    pred = _predict(gold)

    msg.info('Calculating scores ...')

    scores = {'nlpTurk': {}, 'zemberek': {}, 'nltk': {}}
    for m, p in pred.items():
        # sbd scores
        scores[m]['sbd'] = _score(gold['sbd'], p['sbd'], 'sbd')['sbd_per_type']['EOS']
        # lemma scores
        score = _score(gold['conllu'], p['conllu'], 'lemma')
        scores[m]['lemma'] = score['lemma_acc'] if score else None
        # pos scores
        score = _score(gold['conllu'], p['conllu'], 'pos')
        if score:
            gold_labels = set(t for g in gold['conllu'] for t in g['pos'])
            avg = {k.replace('pos_', ''): v for k, v in score.items()
                   if k != 'pos_per_type'}
            score = {k: v for k, v in score['pos_per_type'].items() if k in gold_labels}
            score.update(avg)
        scores[m]['pos'] = score

    FS.to_disk(_create_report(scores, filenames, stats), output_path)

    msg.info(f'Benchmark report saved to `{Path(output_path).resolve()}`')


def _predict(gold: List[Dict[str, List[str]]]) -> Dict[str, List[Dict[str, List[str]]]]:
    """Perform predictions on sentence groups.

    Args:
        gold (List[Dict[str, List[str]]]): List of sentence groups.

    Returns:
        Dict[str, List[Dict[str, List[str]]]]: Predictions.
    """
    msg = Printer()
    predictions = {k: {'conllu': [], 'sbd': []} for k in ('nlpTurk', 'zemberek', 'nltk')}

    # nlptTurk predictions
    msg.info('Executing nlpTurk predictions ...')
    for sents in gold['conllu']:
        data = {'tokens': [], 'lemma': [], 'pos': []}
        for token in nlpturk(' '.join(sents['tokens'])):
            data['tokens'].append(token.text)
            data['lemma'].append(token.lemma)
            data['pos'].append(token.pos)
        predictions['nlpTurk']['conllu'].append(data)
    for sents in gold['sbd']:
        data = {'tokens': [], 'sbd': []}
        for token in nlpturk(' '.join(sents['tokens'])):
            data['tokens'].append(token.text)
            data['sbd'].append('EOS' if token.is_sent_end else 'O')
        # for last token of each sentence group label `sbd` as `O`
        data['sbd'][-1] = 'O'
        predictions['nlpTurk']['sbd'].append(data)

    # zemberek predictions
    msg.info('Executing zemberek predictions ...')
    zemberek = Zemberek()
    for sents in gold['conllu']:
        data = {'tokens': [], 'lemma': [], 'pos': []}
        for sent in zemberek.extract_sents(' '.join(sents['tokens'])):
            for token in zemberek.extract_morphs(sent):
                data['tokens'].append(token['token'])
                data['lemma'].append(token['lemma'])
                data['pos'].append(token['pos'])
        predictions['zemberek']['conllu'].append(data)
    for sents in gold['sbd']:
        data = {'tokens': [], 'sbd': []}
        for sent in zemberek.extract_sents(' '.join(sents['tokens'])):
            for token in zemberek.extract_morphs(sent):
                data['tokens'].append(token['token'])
                data['sbd'].append('O')
            data['sbd'][-1] = 'EOS'
        # for last token of each sentence group label `sbd` as `O`
        data['sbd'][-1] = 'O'
        predictions['zemberek']['sbd'].append(data)

    # nltk predictions
    msg.info('Executing NLTK predictions ...')
    predictions['nltk']['sbd'].extend(_nltk_tokenize(gold['sbd']))

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
        data = {'tokens': [], 'sbd': []}
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
    nlp = spacy.blank('tr')
    nlp.tokenizer = Tokenizer(nlp)

    def make_doc(content):
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
    if not gold or not pred:
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


def _create_report(scores: Dict[str, Any], filenames: str, stats: Dict[str, int]) -> str:
    """Creates benchmark report.

    Args:
        scores (Dict[str, Any]): PRF and accuracy scores per library.
        filenames (str): Benchmark files.
        stats (Dict[str, int]): File statistics (# of sents, # of tokens).

    Returns:
        str: Pretty formatted benchmark report.
    """
    report = [f"{'-'*60}\nBENCHMARK REPORT\n{'-'*60}\n"]
    report.append(f'Repository: https://github.com/nlpturk\n')
    report.append(f'Date:  {datetime.today().strftime("%d/%m/%Y")}')
    report.append(f'Stats: {stats["sents"]} sentences, {stats["tokens"]} tokens')
    if len(filenames) > 1:
        report.append(f'Files: {filenames[0]}')
        for i in range(1, len(filenames)):
            report.append(f'{" "*7}{filenames[i]}')
    else:
        report.append(f'File:  {filenames[0]}')

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
