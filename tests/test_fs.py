import pytest
import json

from nlpturk.fs import FS


conllu_text = '''
# sent_id = bio_9
# text = Fakülteyi bitirenler en uçtan göreve başlıyorlarmış.
1	Fakülteyi	fakülte	NOUN	Noun	Case=Acc|Number=Sing|Person=3	2	obj	_	_
2	bitirenler	bitir	VERB	Verb	Case=Nom|Number=Plur|Person=3|Polarity=Pos|Tense=Pres|VerbForm=Part	6	csubj	_	_
3	en	en	ADV	Adverb	_	4	advmod	_	_
4	uçtan	uç	VERB	Verb	Case=Abl|Number=Sing|Person=3	6	obl	_	_
5	göreve	görev	NOUN	Noun	Case=Dat|Number=Sing|Person=3	6	obj	_	_
6	başlıyorlarmış	başla	VERB	Verb	Aspect=Prog|Evident=Nfh|Number=Plur|Person=3|Polarity=Pos|Tense=Past	0	root	_	SpaceAfter=No
7	.	.	PUNCT	Punc	_	6	punct	_	SpacesAfter=\n

# sent_id = ins_1282
# text = Eşeklerin sırtlarına yüklenmiş sepetlerle taşınırdı üzümler.
1	Eşeklerin	eşek	NOUN	Noun	Case=Gen|Number=Plur|Person=3	2	nmod:poss	_	_
2	sırtlarına	sırt	NOUN	Noun	Case=Dat|Number=Plur|Number[psor]=Sing|Person=3|Person[psor]=3	3	obl	_	_
3	yüklenmiş	yükle	VERB	Verb	Evident=Nfh|Number=Sing|Person=3|Polarity=Pos|Tense=Past|Voice=Pass	4	acl	_	_
4	sepetlerle	sepet	NOUN	Noun	Case=Ins|Number=Plur|Person=3	5	obl	_	_
5	taşınırdı	taşın	VERB	Verb	Aspect=Hab|Evident=Fh|Number=Sing|Person=3|Polarity=Pos|Tense=Pres	0	root	_	_
6	üzümler	üzüm	NOUN	Noun	Case=Nom|Number=Plur|Person=3	5	nsubj	_	SpaceAfter=No
7	.	.	PUNCT	Punc	_	5	punct	_	SpacesAfter=\n
'''


@pytest.fixture(scope='session')
def conllu_file(tmp_path_factory):
    fp = tmp_path_factory.mktemp('conllu') / 'test.conllu'
    with open(fp, 'w', encoding='utf-8') as f:
        f.write(conllu_text)
    return fp


def test_read_conll(conllu_file):
    sents = list(FS.read(conllu_file))
    assert len(sents) == 2
    assert len(sents[0]) == 9
    assert len(sents[0][2]) == 10
    assert len(sents[1][4]) == 10
    assert sents[0][0] == '# sent_id = bio_9'
    assert sents[0][1] == '# text = Fakülteyi bitirenler en uçtan göreve başlıyorlarmış.'
    assert sents[0][2][1] == 'Fakülteyi'
    assert sents[1][1] == '# text = Eşeklerin sırtlarına yüklenmiş sepetlerle taşınırdı üzümler.'
    assert sents[1][3][5] == 'Case=Dat|Number=Plur|Number[psor]=Sing|Person=3|Person[psor]=3'


@pytest.fixture(scope='session')
def json_file(tmp_path_factory):
    content = {'foo': 'foo value', 'bar': 2, 1: 5, 3.14: 'pi'}
    fp = tmp_path_factory.mktemp('json') / 'test.json'
    with open(fp, 'w', encoding='utf-8') as f:
        json.dump(content, f, ensure_ascii=False)
    return fp


def test_append_json(json_file):
    # json dump silently converts all keys to strings
    content = FS.read(json_file)
    assert len(content) == 4
    assert content['foo'] == 'foo value'
    assert content['bar'] == 2
    assert content['1'] == 5
    assert content['3.14'] == 'pi'
    # append new data to json file
    append_data = {'bar': 'updated bar value', '1': 'updated value', 'pi': 3.14}
    FS.to_disk(append_data, json_file, append=True)
    content = FS.read(json_file)
    assert len(content) == 5
    assert content['foo'] == 'foo value'
    assert content['bar'] == 'updated bar value'
    assert content['1'] == 'updated value'
    assert content['3.14'] == 'pi'
    assert content['pi'] == 3.14
    # attemp to append a list to json dictionary raises ValueError
    with pytest.raises(ValueError):
        FS.to_disk([1, {'a': 2}], json_file, append=True)
