import pytest

from nlpturk.training.preprocess import _parse_conllu


conllu_text = '''
# sent_id = mst-0025
# text = Kot pantolonla gelen hiçbir çocuk yoktu.
1	Kot	kot	ADJ	Adj	_	2	amod	_	_
2	pantolonla	pantolon	NOUN	Noun	Case=Ins|Number=Sing|Person=3	3	obl	_	_
3	gelen	gel	VERB	Verb	Aspect=Perf|Mood=Ind|Polarity=Pos|Tense=Pres|VerbForm=Part	5	acl	_	_
4	hiçbir	hiçbir	DET	Det	_	5	det	_	_
5	çocuk	çocuk	NOUN	Noun	Case=Nom|Number=Sing|Person=3	6	nsubj	_	_
6-7	yoktu	_	_	_	_	_	_	_	SpaceAfter=No
6	yok	yok	ADV	Adverb	_	0	root	_	_
7	tu	i	AUX	Zero	Aspect=Perf|Mood=Ind|Number=Sing|Person=3|Tense=Past	6	cop	_	_
8	.	.	PUNCT	Punc	_	6	punct	_	_

# sent_id = mst-0336
# text = AB ve Irak riskleri korkutuyor.
1	AB	Ab	NOUN	Abr	Abbr=Yes|Case=Nom|Number=Sing|Person=3	4	nmod:poss	_	_
2	ve	ve	CCONJ	Conj	_	3	cc	_	_
3	Irak	Irak	PROPN	Prop	Case=Nom|Number=Sing|Person=3	1	conj	_	_
4	riskleri	risk	NOUN	Noun	Case=Nom|Number=Sing|Number[psor]=Plur|Person=3|Person[psor]=3	5	nsubj	_	_
5	korkutuyor	korkut	VERB	Verb	Aspect=Prog|Mood=Ind|Number=Sing|Person=3|Polarity=Pos|Polite=Infm|Tense=Pres	0	root	_	SpaceAfter=No
6	.	.	PUNCT	Punc	_	5	punct	_	_

# sent_id = mst-2462
# text = Duyarlı kılıyor.
1	Duyarlı	duyarlı	ADJ	Adj	_	2	obj	_	_
2	kılıyor	kıl	VERB	Verb	Aspect=Prog|Mood=Ind|Number=Sing|Person=3|Polarity=Pos|Polite=Infm|Tense=Pres	0	root	_	SpaceAfter=No
3	.	.	PUNCT	Punc	_	2	punct	_	_
'''


@pytest.fixture(scope='session')
def conllu_file(tmp_path_factory):
    fp = tmp_path_factory.mktemp('conllu') / 'test.conllu'
    with open(fp, 'w', encoding='utf-8') as f:
        f.write(conllu_text)
    return fp


def test_parse_conll(conllu_file):
    sents = _parse_conllu(conllu_file, min_tokens=5)
    assert len(sents) == 2
    assert sents[0]['words'] == ['Kot', 'pantolonla', 'gelen', 'hiçbir',
                                 'çocuk', 'yoktu', '.']
    assert sents[0]['poses'] == ['ADJ', 'NOUN', 'VERB', 'DET', 'NOUN', 'ADV', 'PUNCT']
    assert sents[0]['morphs'][2] == 'Aspect=Perf|Mood=Ind|Polarity=Pos|Tense=Pres|VerbForm=Part'
    assert sents[1]['words'] == ['AB', 've', 'Irak', 'riskleri', 'korkutuyor', '.']
    assert sents[1]['lemmas'] == ['AB', 've', 'Irak', 'risk', 'korkut', '.']
    assert sents[1]['poses'] == ['NOUN', 'CCONJ', 'PROPN', 'NOUN', 'VERB', 'PUNCT']
