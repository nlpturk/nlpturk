import re
from itertools import islice
from typing import Callable, Optional, Iterable, Tuple, Dict, List, Any

import numpy as np
from thinc.api import Model, SequenceCategoricalCrossentropy, Config
from spacy.tokens import Token
from spacy.tokens.doc import Doc
from spacy.pipeline.tagger import Tagger
from spacy.language import Language
from spacy.vocab import Vocab
from spacy.training.example import Example
from spacy.errors import Errors
from spacy.scorer import PRFScore
from spacy.training import validate_examples, validate_get_examples
from spacy.util import registry
from spacy import util


default_model_config = """
[model]
@architectures = "spacy.Tagger.v2"

[model.tok2vec]
@architectures = "spacy.HashEmbedCNN.v2"
pretrained_vectors = null
width = 96
depth = 4
embed_size = 2000
window_size = 1
maxout_pieces = 3
subword_features = true
"""
DEFAULT_SBD_MODEL = Config().from_str(default_model_config)["model"]

Token.set_extension("sent_end", default=False, force=True)
Token.set_extension("sbd_tag", default=False, force=True)
Token.set_extension("sbd_tag_", default=False, force=True)


@Language.factory(
    "sbd",
    assigns=["token._.sent_end", "token._.sbd_tag", "token._.sbd_tag_"],
    default_config={"model": DEFAULT_SBD_MODEL, "overwrite": False,
                    "scorer": {"@scorers": "spacy.sbd_scorer.v1"}, "neg_prefix": "!"},
    default_score_weights={"sbd_f": 1.0, "sbd_p": 0.0, "sbd_r": 0.0}
)
def make_sbd(
    nlp: Language,
    name: str,
    model: Model,
    overwrite: bool,
    scorer: Optional[Callable],
    neg_prefix: str,
):
    """Construct a sentence boundary detection component.
    """
    return SentenceBoundaryDetector(nlp.vocab, model, name, overwrite=overwrite,
                                    scorer=scorer, neg_prefix=neg_prefix)


def sbd_score(examples, **kwargs):
    micro_prf = PRFScore()
    for example in examples:
        gold_doc = example.reference
        pred_doc = example.predicted
        align = example.alignment
        gold_tags = set()
        for gold_i, token in enumerate(gold_doc):
            gold_tags.add((gold_i, token._.sbd_tag))
        pred_tags = set()
        for token in pred_doc:
            if token.orth_.isspace():
                continue
            if align.x2y.lengths[token.i] == 1:
                gold_i = align.x2y[token.i][0]
                pred_tags.add((gold_i, token._.sbd_tag))
        micro_prf.score_set(pred_tags, gold_tags)
    return {"sbd_p": micro_prf.precision, "sbd_r": micro_prf.recall,
            "sbd_f": micro_prf.fscore}


@registry.scorers("spacy.sbd_scorer.v1")
def make_sbd_scorer():
    return sbd_score


class SentenceBoundaryDetector(Tagger):
    """Pipeline component for sentence boundary detection.
    """

    def __init__(
        self,
        vocab: Vocab,
        model: Model,
        name: str = "sbd",
        *,
        overwrite: bool = False,
        scorer: Optional[Callable] = sbd_score,
        neg_prefix: str = "!",
    ):
        """Initialize a sentence boundary detector.

        Args:
            vocab (Vocab): The shared vocabulary.
            model (Model): The Thinc Model powering the pipeline component.
            name (str): The component instance name, used to add entries to the
                losses during training.
            scorer (Optional[Callable]): The scoring method.
        """
        self.vocab = vocab
        self.model = model
        self.name = name
        self._rehearsal_model = None
        cfg = {"labels": [], "overwrite": overwrite, "neg_prefix": neg_prefix}
        self.cfg = dict(sorted(cfg.items()))
        self.scorer = scorer

    def get_aligned(self, example: Example) -> List[Any]:
        """
        Args:
            example (Example): Example object.

        Returns:
            List[Any]: Aligned array for a token attribute.
        """
        align = example.alignment.x2y
        vocab = example.reference.vocab
        gold_values = np.array([t._.sbd_tag for t in example.reference], dtype=np.uint64)
        output = [None] * len(example.predicted)
        for token in example.predicted:
            values = gold_values[align[token.i]]
            values = values.ravel()
            if len(values) == 0:
                output[token.i] = None
            elif len(values) == 1:
                output[token.i] = values[0]
            elif len(set(list(values))) == 1:
                # If all aligned tokens have the same value, use it.
                output[token.i] = values[0]
            else:
                output[token.i] = None
        return [vocab.strings[o] if o is not None else o for o in output]

    def set_annotations(self, docs: Iterable[Doc], batch_tag_ids):
        """Modify a batch of documents, using pre-computed scores.

        Args:
            docs (Iterable[Doc]): The documents to modify.
            batch_tag_ids: The IDs to set, produced by Tagger.predict.
        """
        if isinstance(docs, Doc):
            docs = [docs]
        labels = self.labels
        for i, doc in enumerate(docs):
            doc_tag_ids = batch_tag_ids[i]
            if hasattr(doc_tag_ids, "get"):
                doc_tag_ids = doc_tag_ids.get()
            eos_token = None
            for j, token in enumerate(doc):
                if eos_token:
                    if (not eos_token.whitespace_ and not re.search(r'[^\W_]', token.text)) \
                            or re.search(r'^[.?!:;)}\]]', token.text):
                        eos_token = token
                    else:
                        token.sent_start = True
                        eos_token._.sent_end = True
                        eos_token = None
                if token._.sbd_tag == 0 or self.cfg["overwrite"]:
                    token._.sbd_tag = self.vocab.strings[labels[doc_tag_ids[j]]]
                    token._.sbd_tag_ = labels[doc_tag_ids[j]]
                    if token._.sbd_tag_ == 'EOS':
                        eos_token = token
            token._.sent_end = True if eos_token else False

    def get_loss(self, examples: Iterable[Example], scores) -> Tuple[float, float]:
        """Find the loss and gradient of loss for the batch of documents and
        their predicted scores.

        Args:
            examples (Iterable[Example]): The batch of examples.
            scores: Scores representing the model's predictions.

        Returns:
            (Tuple[float, float]): The loss and the gradient.
        """
        validate_examples(examples, "SentenceBoundaryDetector.get_loss")
        loss_func = SequenceCategoricalCrossentropy(names=self.labels, normalize=False,
                                                    neg_prefix=self.cfg["neg_prefix"])
        # Convert empty tag "" to missing value None so that both misaligned tokens
        # and tokens with missing annotation have the default missing value None.
        truths = []
        for eg in examples:
            eg_truths = [tag if tag != "" else None
                         for tag in self.get_aligned(eg)]
            truths.append(eg_truths)
        d_scores, loss = loss_func(scores, truths)
        if self.model.ops.xp.isnan(loss):
            raise ValueError(Errors.E910.format(name=self.name))
        return float(loss), d_scores

    def initialize(
        self,
        get_examples: Callable[[], Iterable[Example]],
        *,
        nlp: Language = None,
        labels=None
    ) -> None:
        """Initialize the pipe for training, using a representative set
        of data examples.

        Args:
            get_examples (Callable[[], Iterable[Example]]): Function that
                returns a representative sample of gold-standard Example objects.
            nlp (Language): The current nlp object the component is part of.
            labels: The labels to add to the component, typically generated by the
                `init labels` command. If no labels are provided, the get_examples
                callback is used to extract the labels from the data.
        """
        validate_get_examples(
            get_examples, "SentenceBoundaryDetector.initialize")
        util.check_lexeme_norms(self.vocab, "sbd")
        if labels is not None:
            for tag in labels:
                self.add_label(tag)
        else:
            tags = set()
            for example in get_examples():
                for token in example.y:
                    if token._.sbd_tag_:
                        tags.add(token._.sbd_tag_)
            for tag in sorted(tags):
                self.add_label(tag)
        doc_sample = []
        label_sample = []
        for example in islice(get_examples(), 10):
            doc_sample.append(example.x)
            gold_tags = self.get_aligned(example)
            gold_array = [[1.0 if tag == gold_tag else 0.0 for tag in self.labels]
                          for gold_tag in gold_tags]
            label_sample.append(self.model.ops.asarray(
                gold_array, dtype="float32"))
        self._require_labels()
        assert len(doc_sample) > 0, Errors.E923.format(name=self.name)
        assert len(label_sample) > 0, Errors.E923.format(name=self.name)
        self.model.initialize(X=doc_sample, Y=label_sample)
