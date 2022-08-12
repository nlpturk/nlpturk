import numpy as np
from spacy.tokens.doc import Doc
from spacy.tokens.span import Span
from spacy.tokens import Token as Token_

from .utils import lower, islower, isupper, istitle


class Token:
    """Class that encapsulates the spaCy Token object. Modifies and hides some token attributes.
    The spaCy Token object is available via `_token` attribute.  
    """

    def __init__(self, token: Token_):
        """
        Args:
            token (Token_): spaCy Token object.
            offsets: Tuple[int, str]: The character offset and trailing whitespaces 
                of the token within the document. 
        """
        self._token = token
        self.i = token.i

    def __len__(self):
        """The number of unicode characters in the token.
        """
        return self._token.__len__()

    def __unicode__(self):
        return self._token.__unicode__()

    def __bytes__(self):
        return self._token.__bytes__()

    def __str__(self):
        return self._token.__str__()

    def __repr__(self):
        return self._token.__repr__()

    @property
    def idx(self):
        """Returns:
            int: The character offset of the token within the document.
        """
        # set aligned token offset
        return self._token.doc._.idx[self.i]

    @property
    def pos(self):
        """
        Returns:
            str: Coarse-grained part-of-speech tag.
        """
        return self._token.tag_

    @property
    def lemma(self):
        """
        Returns:
            str: The token lemma.
        """
        return lower(self._token.lemma_)

    @property
    def is_sent_end(self):
        """
        Returns:
            bool: Whether the token ends a sentence.
        """
        return self._token._.sent_end

    @property
    def is_sent_start(self):
        """
        Returns:
            bool: Whether the token starts a sentence.
        """
        return bool(self._token.is_sent_start)

    @property
    def is_lower(self):
        """
        Returns:
            bool: Whether the token is in lowercase.
        """
        return islower(self._token.text)

    @property
    def is_upper(self):
        """
        Returns:
            bool: Whether the token is in uppercase.
        """
        return isupper(self._token.text)

    @property
    def is_title(self):
        """
        Returns:
            bool: Whether the token is in titlecase.
        """
        return istitle(self._token.text)

    @property
    def is_stop_word(self):
        """
        Returns:
            bool: Whether the token is a stop word.
        """
        return self._token.is_stop

    @property
    def text(self):
        """
        Returns:
            str: The text of the token.
        """
        return self._token.text

    @property
    def text_with_ws(self):
        """
        Returns:
            str: The text of the token with trailing whitespaces if exist.
        """
        tws = self.text + self._token.doc._.ws[self.i]
        return tws if self.i > 0 else self._token.doc._.lws + tws

    @property
    def is_alpha(self):
        """
        Returns:
            bool: Whether the token consists of alpha characters.
        """
        return self._token.is_alpha

    @property
    def is_ascii(self):
        """
        Returns:
            bool: Whether the token consists of ASCII characters.
        """
        return self._token.is_ascii

    @property
    def is_bracket(self):
        """
        Returns:
            bool: Whether the token is a bracket.
        """
        return self._token.is_bracket

    @property
    def is_currency(self):
        """
        Returns:
            bool: Whether the token is a currency symbol.
        """
        return self._token.is_currency

    @property
    def is_digit(self):
        """
        Returns:
            bool: Whether the token consists of digits.
        """
        return self._token.is_digit

    @property
    def is_punct(self):
        """
        Returns:
            bool: Whether the token is punctuation.
        """
        return self._token.is_punct

    @property
    def is_quote(self):
        """
        Returns:
            bool: Whether the token is a quotation mark.
        """
        return self._token.is_quote

    @property
    def like_email(self):
        """
        Returns:
            bool: Whether the token resembles an email address.
        """
        return self._token.like_email

    @property
    def like_num(self):
        """
        Returns:
            bool: Whether the token resembles a number.
        """
        return self._token.like_num

    @property
    def like_url(self):
        """
        Returns:
            bool: Whether the token resembles a URL.
        """
        return self._token.like_url


class Sent:
    """Class that encapsulates the spaCy Span object. The spaCy Span object is available 
    via `_span` attribute.
    """

    def __init__(self, span: Span) -> None:
        """
        Args:
            span (Span): spaCy Span object.
        """
        self._span = span

    def __iter__(self):
        """Iterate over the tokens in the document.
        """
        for token in self._span:
            yield Token(token)

    def __getitem__(self, i: int) -> Token:
        """Return token at index `i`.

        Args:
            i (int): Index of the token.

        Returns:
            Token: Token object.
        """
        if not isinstance(i, int):
            raise ValueError('The attribute value should be an integer.')
        return Token(self._span.__getitem__(i))

    def __len__(self):
        """Return the number of tokens in the sentence.
        """
        return self._span.__len__()

    def __repr__(self):
        return self.text

    @property
    def text(self):
        """
        Returns:
            str: The text of the sentence.
        """
        return self.text_with_ws.rstrip()

    @property
    def text_with_ws(self):
        """
        Returns:
            str: The text of the sentence with trailing whitespaces if exist.
        """
        return ''.join(Token(t).text_with_ws for t in self._span)

    @property
    def start(self):
        """
        Returns:
            int: The index of the first token of the sentence. 
        """
        return self._span.start

    @property
    def end(self):
        """
        Returns:
            int: The index of the first token after the sentence. 
        """
        return self._span.end


class Document:
    """Class that encapsulates the spaCy Doc object. The spaCy Doc object is available 
    via `_doc` attribute.
    """

    def __init__(self, doc: Doc) -> None:
        """
        Args:
            doc (Doc): spaCy Doc object.
        """
        self._doc = doc

    def __iter__(self):
        """Iterate over the tokens in the document.
        """
        for token in self._doc:
            yield Token(token)

    def __getitem__(self, i: int) -> Token:
        """Return token at index `i`.

        Args:
            i (int): Index of the token.

        Returns:
            Token: Token object.
        """
        if not isinstance(i, int):
            raise ValueError('The attribute value should be an integer.')
        return Token(self._doc.__getitem__(i))

    def __len__(self):
        """Return the number of tokens in the document.
        """
        return self._doc.__len__()

    def __unicode__(self):
        return self.text

    def __bytes__(self):
        return self.text.encode('utf-8')

    def __str__(self):
        return self.__unicode__()

    def __repr__(self):
        return self.__str__()

    @property
    def sents(self):
        """Iterate over the sentences in the document.
        """
        if not self._doc.has_annotation('SENT_START'):
            yield Sent(Span(self._doc, 0, len(self._doc)))
        else:
            for sent in self._doc.sents:
                yield Sent(sent)

    @property
    def text(self):
        """
        Returns:
            str: The string representation of the document text. 
        """
        return ''.join(Token(t).text_with_ws for t in self._doc)
