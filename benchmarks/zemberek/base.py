from jpype import JClass
from wasabi import Printer

from ._util import start_jvm, POS_TAGS


class Zemberek:
    """Wrapper class for `zemberek` library.
    """

    def __init__(self) -> None:
        # start a Java Virtual Machine
        start_jvm()
        # zemberek TurkishSentenceExtractor object
        self.sbd = JClass('zemberek.tokenization.TurkishSentenceExtractor').DEFAULT
        # zemberek TurkishMorphology object
        Printer().info('Initializing `zemberek.morphology.TurkishMorphology` ...')
        self.morph = JClass('zemberek.morphology.TurkishMorphology').createWithDefaults()

    def extract_sents(self, text: str):
        """
        Zemberek sentence boundary detection.

        Args:
            text (str): Text to extract sentences.
        """
        return [str(s) for s in self.sbd.fromParagraph(text)]

    def extract_morphs(self, sent: str):
        """
        Zemberek lemmatization and POS tag detection.

        Args:
            sent (str): Sentence to find POS tags on and apply lemmatization.
        """
        pred, chars = [], ''.join(sent.split()).replace("''", '❠')
        for a in self.morph.analyzeAndDisambiguate(sent):
            token = chars[:len(str(a.getWordAnalysis().getInput()))]
            morph = a.getBestAnalysis()
            lemma = str(morph.getLemmas()[0])
            lemma = token if lemma == 'UNK' else lemma
            pos = str(morph.getPos().shortForm)
            pos = POS_TAGS[pos] if pos in POS_TAGS else ''
            pred.append({'token': token.replace('❠', "''"), 'lemma': lemma, 'pos': pos})
            chars = chars[len(token):]
        return pred
