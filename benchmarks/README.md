# nlpTurk - Benchmarks

**Files:**
[tr_penn-ud-test](https://raw.githubusercontent.com/UniversalDependencies/UD_Turkish-Penn/master/tr_penn-ud-test.conllu),
[tr_atis-ud-test](https://raw.githubusercontent.com/UniversalDependencies/UD_Turkish-Atis/master/tr_atis-ud-test.conllu),
[tr_framenet-ud-test](https://raw.githubusercontent.com/UniversalDependencies/UD_Turkish-FrameNet/master/tr_framenet-ud-test.conllu),
[tr_kenet-ud-test](https://raw.githubusercontent.com/UniversalDependencies/UD_Turkish-Kenet/master/tr_kenet-ud-test.conllu)
<br/>
**Stats:** 3147 sentences, 33411 tokens

## Sentence Segmentation

|              | precision | recall    | f1-score  | 
| :----------- | :-------: | :-------: | :-------: | 
| **nlpTurk**  | **97.84** | **95.83** | **96.82** |  
| **zemberek** |   97.11   |   78.32   |   86.71   |   
| **NLTK**     |   96.41   |   77.75   |   86.08   |

## Lemmatization

|              | accuracy  | 
| :----------- | :-------: | 
| **nlpTurk**  | **96.87** |  
| **zemberek** |   90.49   |  

## POS Tagging

### Micro Average

|              | precision | recall    | f1-score  | 
| :----------- | :-------: | :-------: | :-------: | 
| **nlpTurk**  | **95.75** | **96.26** | **96.01** |  
| **zemberek** |   83.87   |   83.95   |   83.91   | 


### ADJ \[adjective\]

|              | precision | recall    | f1-score  | 
| :----------- | :-------: | :-------: | :-------: | 
| **nlpTurk**  | **94.58** | **93.34** | **93.95** |  
| **zemberek** |   86.53   |   75.78   |   80.80   | 


### ADP \[adposition\]

|              | precision | recall    | f1-score  | 
| :----------- | :-------: | :-------: | :-------: | 
| **nlpTurk**  | **94.97** | **96.90** | **95.93** |  
| **zemberek** |   86.50   |   89.07   |   87.77   | 


### ADV \[adverb\]

|              | precision | recall    | f1-score  | 
| :----------- | :-------: | :-------: | :-------: | 
| **nlpTurk**  | **93.76** | **92.94** | **93.35** |  
| **zemberek** |   91.22   |   71.82   |   80.37   | 


### AUX \[auxiliary\]

|              | precision | recall    | f1-score  | 
| :----------- | :-------: | :-------: | :-------: | 
| **nlpTurk**  |   94.83   | **99.40** | **97.06** |  
| **zemberek** |**100.00** |   62.05   |   76.58   | 


### CCONJ \[coordinating conjunction\]

|              | precision | recall    | f1-score  | 
| :----------- | :-------: | :-------: | :-------: | 
| **nlpTurk**  | **95.99** | **98.59** | **97.27** |  
| **zemberek** |   85.64   |   94.82   |   89.99   | 


### DET \[determiner\]

|              | precision | recall    | f1-score  | 
| :----------- | :-------: | :-------: | :-------: | 
| **nlpTurk**  | **96.49** |   97.74   | **97.11** |  
| **zemberek** |   90.29   | **99.11** |   94.49   | 


### INTJ \[interjection\]

|              | precision | recall    | f1-score  | 
| :----------- | :-------: | :-------: | :-------: | 
| **nlpTurk**  | **93.94** | **79.49** | **86.11** |  
| **zemberek** |   60.53   |   58.97   |   59.74   | 


### NOUN \[noun\]

|              | precision | recall    | f1-score  | 
| :----------- | :-------: | :-------: | :-------: | 
| **nlpTurk**  | **97.92** | **96.06** | **96.98** |  
| **zemberek** |   77.48   |   95.20   |   85.43   | 


### NUM \[numeral\]

|              | precision | recall    | f1-score  | 
| :----------- | :-------: | :-------: | :-------: | 
| **nlpTurk**  | **95.25** | **97.86** | **96.54** |  
| **zemberek** |   89.89   |   89.62   |   89.76   | 


### PRON \[pronoun\]

|              | precision | recall    | f1-score  | 
| :----------- | :-------: | :-------: | :-------: | 
| **nlpTurk**  | **90.66** | **92.04** | **91.34** |  
| **zemberek** |   89.35   |   85.48   |   87.37   | 


### PROPN \[proper noun\]

|              | precision | recall    | f1-score  | 
| :----------- | :-------: | :-------: | :-------: | 
| **nlpTurk**  | **92.66** | **95.68** | **94.15** |  
| **zemberek** |   0.00    |   0.00    |   0.00    | 


### PUNCT \[punctuation\]

|              | precision | recall    | f1-score  | 
| :----------- | :-------: | :-------: | :-------: | 
| **nlpTurk**  | **98.67** | **99.34** | **99.01** |  
| **zemberek** |   97.61   |   99.05   |   98.32   | 


### SCONJ \[subordinating conjunction\]

|              | precision | recall    | f1-score  | 
| :----------- | :-------: | :-------: | :-------: | 
| **nlpTurk**  | **100.00**| **97.83** | **98.90** |  
| **zemberek** |   0.00    |   0.00    |   0.00    | 


### VERB \[verb\]

|              | precision | recall    | f1-score  | 
| :----------- | :-------: | :-------: | :-------: | 
| **nlpTurk**  | **92.43** | **98.40** | **95.32** |  
| **zemberek** |   89.48   |   96.24   |   92.74   | 


### X \[other\]

|              | precision | recall    | f1-score  | 
| :----------- | :-------: | :-------: | :-------: | 
| **nlpTurk**  | **26.15** | **44.74** | **33.01** |  
| **zemberek** |   0.00    |   0.00    |   0.00    | 
