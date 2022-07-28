import spacy
from nlpturk.pipeline.tokenizer import Tokenizer


def test_url_match():
    valid_urls = ['nlpturk.com.tr', 'http://nlpturk.net', 'www.nlpturk.ai']
    invalid_urls = ['nlpturk.fake', 'http:nlpturk.net', 'www.nlpturk', 'nlpturk.Com']

    nlp = spacy.blank('tr')
    nlp.tokenizer = Tokenizer(nlp.vocab)
    # valid urls
    for url in valid_urls:
        tokens = [t.text for t in nlp(url)]
        assert len(tokens) == 1
    # invalid urls
    for url in invalid_urls:
        tokens = [t.text for t in nlp(url)]
        assert len(tokens) > 1
