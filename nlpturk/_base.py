import re
import sys
import itertools
import warnings
from pathlib import Path
from importlib.util import find_spec

import spacy
from spacy import util
from spacy.tokens.doc import Doc
from wasabi import Printer

from . import pkg
from .pipeline.tokenizer import Tokenizer
from .pipeline import sbd
from .doc import Document


class _M(sys.modules[__name__].__class__):
    """Class that makes the nlpturk package a callable module.
    """

    def __call__(self, text: str, trf: bool = False) -> Document:
        """Makes the nlpturk package callable.

        Usage: 
            import nlpturk
            doc = nlpturk(some_text)

            # iterate over tokens
            for token in doc:
                print(token.text, token.lemma, token.pos, token.is_sent_end)

            # or get tokens by token ids
            token = doc[i]

            # iterate over sentences
            for sent in doc.sents:
                print(sent.text, sent.vector)
                for token in sent:
                    print(token.text, token.lemma, token.pos, token.vector)

        Args:
            text (str): Text to be processed.
            trf (bool, optional): Whether to use transformer model or not. Uses tok2vec model
                as default. Transformer model is slightly more accurate with the cost of 
                higher inference time. For model comparisons and usage details visit project's page.
                https://github.com/nlpturk/nlpturk

        Returns:
            Document: Document object.
        """
        if not hasattr(self, '_nlp'):
            warnings.filterwarnings('ignore')
            try:
                model_path = Path(find_spec(pkg.__model__).origin).parent
            except AttributeError:
                model_path = self._download()
            self._nlp = spacy.load(model_path)
            self._nlp.tokenizer = Tokenizer(self._nlp)

        return Document(self._get_doc(text))

    def _get_doc(self, text) -> Doc:
        """Removes whitespace tokens returned from Tokenizer and aligns character offsets
        of the tokens. 

        Args:
            text (_type_): Raw text.

        Returns:
            Doc: spaCy Doc object.
        """
        # grab trailing whitespaces for each word
        idx, ws = [0], []
        for m in re.finditer(r'\s+', text):
            if m.start(0) > 0:
                idx.append(m.end(0))
                ws.append(m.group(0))

        # grab leading whitespaces from text
        lws = ''.join(itertools.takewhile(str.isspace, text))
        idx[0] = len(lws) if lws else 0

        doc = self._nlp(' '.join(text.split()))

        # set aligned offsets and trailing whitespaces for each token
        tokens, i = [], idx[0]
        for t in doc:
            if t.whitespace_:
                tokens.append((i, ws[0]))
                idx.pop(0)
                ws.pop(0)
                i = idx[0]
            else:
                tokens.append((i, ''))
                i += len(t.text)

        # add trailing whitespaces to last token, if exist
        tokens[-1] = (tokens[-1][0], ws[0]) if ws else tokens[-1]

        doc.set_extension('idx', default=[t[0] for t in tokens], force=True)
        doc.set_extension('ws', default=[t[1] for t in tokens], force=True)
        doc.set_extension('lws', default=lws, force=True)

        return doc

    def _download(self) -> None:
        """Downloads nlpTurk model.

        Args:
            model (str): nlpTurk model name
        """
        msg = Printer()
        msg.info(f'Model `{pkg.__model__}` not found! Downloading ...')
        cmd = [sys.executable, "-m", "pip", "install"] + [pkg.__download_url__]
        try:
            util.run_command(cmd)
            model_path = Path(find_spec(pkg.__model__).origin).parent
            msg.good(f'Model `{pkg.__model__}` is downloaded successfully.')
            return model_path
        except:
            msg.fail('Connection to server is failed.')
            msg.fail('Please try again later and make sure your Internet connection is on.')
            raise
