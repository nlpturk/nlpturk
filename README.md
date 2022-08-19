# nlpTurk - Turkish NLP library

nlpTurk is an open source Turkish NLP library consisting of machine learning based sentence boundary detection, lemmatization and POS tagging models.

## Installation & Usage

nlpTurk can be installed from [PyPI](https://pypi.org/project/nlpturk/). 
 
```bash
pip install nlpturk
```

nlpTurk offers a simple API to extract sentences, lemmas and POS tags.

```python
import nlpturk

text = "Sosyal medya hayatımıza hızlı girdi.ama yazım kurallarına dikkat eden pek yok :)"
doc = nlpturk(text)

# iterate over tokens
for token in doc:
    print(f"token: {token.text}, lemma: {token.lemma}, pos: {token.pos}")

"""
Prints:
  token: Sosyal, lemma: sosyal, pos: ADJ
  token: medya, lemma: medya, pos: NOUN
  ...
"""

# or get tokens by token ids
token = doc[5]
print(f"token: {token.text}, sent_start: {token.is_sent_start}, sent_end: {token.is_sent_end}")
token = doc[6]
print(f"token: {token.text}, sent_start: {token.is_sent_start}, sent_end: {token.is_sent_end}")

"""
Prints:
  token: ., sent_start: False, sent_end: True
  token: ama, sent_start: True, sent_end: False
"""

# iterate over sentences
for i, sent in enumerate(doc.sents):
    print(f"sentence #{i+1}: {sent.text}")
    for token in sent:
        print(f"  token: {token.text}, lemma: {token.lemma}, pos: {token.pos}")

"""
Prints:
  sentence #1: Sosyal medya hayatımıza hızlı girdi.
    token: Sosyal, lemma: sosyal, pos: ADJ
    ...
  sentence #2: ama yazım kurallarına dikkat eden pek yok :)
    token: ama, lemma: ama, pos: CCONJ
    ...
"""
```

## Performance

The evaluation was performed on test dataset. Detailed evaluation and benchmarking results can be found [here](https://github.com/nlpturk/nlpturk/blob/master/benchmarks).

|                        | accuracy | precision | recall | f1-score | 
| :--------------------- | :------: | :-------: | :----: | :------: | 
| **Sentence Segmenter** |    -     |   97.84   |  95.83 |  96.82   |  
| **POS Tagger**         |    -     |   95.75   |  96.26 |  96.01   |   
| **Lemmatizer**         |  96.87   |     -     |    -   |    -     |

<br/>You can perform benchmarking on your own dataset.

```bash
git clone https://github.com/nlpturk/nlpturk.git
cd nlpturk
pip install -r requirements.txt
python -m nlpturk benchmark --data_path path/to/data --output_path path/to/output
```