import os
from pathlib import Path

from jpype import getDefaultJVMPath, startJVM, isJVMStarted


# Universal Dependencies equivalents of zemberek POS tags
POS_TAGS = {
    'Noun': 'NOUN', 'Adj': 'ADJ', 'Num': 'NUM', 'Adv': 'ADV', 'Det': 'DET',
    'Pron': 'PRON', 'Interj': 'INTJ', 'Punc': 'PUNCT', 'Conj': 'CCONJ',
    'Postp': 'ADP', 'Verb': 'VERB', 'Ques': 'AUX'
}


def start_jvm():
    """Starts Java Virtual Machine.
    """
    # make sure Java is installed
    if os.getenv('JAVA_HOME') is None:
        raise EnvironmentError(
            'Install Java and make sure the `JAVA_HOME` environment variable is set.'
        )
    # start a Java Virtual Machine if not already started
    if not isJVMStarted():
        startJVM(
            jvmpath=getDefaultJVMPath(),
            classpath=os.path.join(Path(__file__).resolve().parent,
                                   'bin', 'zemberek-full.jar'),
            convertStrings=False,
            interrupt=True
        )
