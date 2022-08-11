import re

import spacy
from spacy.tokens import Doc
from spacy.language import Language
from .tld import TLD


class Tokenizer:
    """Class for Turkish text tokenization. Applies a more robust url match pattern based
    on a list of valid top level domains.
    """

    def __init__(self, nlp: Language):
        self.nlp = spacy.blank('tr')
        self.nlp.max_length = nlp.max_length
        self.vocab = nlp.vocab

    def _is_url(self, text: str) -> bool:
        """Checks whether the string is a valid url or not.
        https://gist.github.com/gruber/8891611
        """
        tld = '|'.join(TLD)
        pattern = re.compile(
            fr'(?i)\b((?:https?:(?:/{{1,3}}|[a-z0-9%])|[a-z0-9.\-]+[.](?:{tld})/)'
            r'(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+'
            r'(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{{}};:\'"'
            fr'.,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:{tld})\b/?(?!@)))'
        )
        return bool(pattern.match(text))

    def __call__(self, text):
        words, spaces = [], []
        for token in self.nlp(text):
            if token.like_url and not self._is_url(token.text):
                tokens = token.text.replace('.', ' . ').split()
                words.extend(tokens)
                spaces.extend([False] * len(tokens))
                spaces[-1] = bool(token.whitespace_)
            else:
                words.append(token.text)
                spaces.append(bool(token.whitespace_))
        return Doc(self.vocab, words=words, spaces=spaces)
