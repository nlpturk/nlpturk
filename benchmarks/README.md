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
| **nlpTurk**  | **97.69** | **97.18** | **97.43** |  
| **zemberek** |   97.11   |   78.32   |   86.71   |   
| **NLTK**     |   96.41   |   77.75   |   86.08   |

## Lemmatization

|              | accuracy  | 
| :----------- | :-------: | 
| **nlpTurk**  | **96.58** |  
| **zemberek** |   90.49   |  

## POS Tagging

### Micro Average

|              | precision | recall    | f1-score  | 
| :----------- | :-------: | :-------: | :-------: | 
| **nlpTurk**  | **95.87** | **96.38** | **96.12** |  
| **zemberek** |   83.87   |   83.95   |   83.91   | 


### ADJ \[adjective\]

|              | precision | recall    | f1-score  | 
| :----------- | :-------: | :-------: | :-------: | 
| **nlpTurk**  | **94.24** | **93.86** | **94.05** |  
| **zemberek** |   86.53   |   75.78   |   80.80   | 


### ADP \[adposition\]

|              | precision | recall    | f1-score  | 
| :----------- | :-------: | :-------: | :-------: | 
| **nlpTurk**  | **94.31** | **96.22** | **95.26** |  
| **zemberek** |   86.50   |   89.07   |   87.77   | 


### ADV \[adverb\]

|              | precision | recall    | f1-score  | 
| :----------- | :-------: | :-------: | :-------: | 
| **nlpTurk**  | **93.27** | **93.04** | **93.15** |  
| **zemberek** |   91.22   |   71.82   |   80.37   | 


### AUX \[auxiliary\]

|              | precision | recall    | f1-score  | 
| :----------- | :-------: | :-------: | :-------: | 
| **nlpTurk**  |   97.65   |**100.00** | **98.81** |  
| **zemberek** |**100.00** |   62.05   |   76.58   | 


### CCONJ \[coordinating conjunction\]

|              | precision | recall    | f1-score  | 
| :----------- | :-------: | :-------: | :-------: | 
| **nlpTurk**  | **94.69** | **98.70** | **96.66** |  
| **zemberek** |   85.64   |   94.82   |   89.99   | 


### DET \[determiner\]

|              | precision | recall    | f1-score  | 
| :----------- | :-------: | :-------: | :-------: | 
| **nlpTurk**  | **95.36** |   98.36   | **96.84** |  
| **zemberek** |   90.29   | **99.11** |   94.49   | 


### INTJ \[interjection\]

|              | precision | recall    | f1-score  | 
| :----------- | :-------: | :-------: | :-------: | 
| **nlpTurk**  | **91.89** | **87.18** | **89.47** |  
| **zemberek** |   60.53   |   58.97   |   59.74   | 


### NOUN \[noun\]

|              | precision | recall    | f1-score  | 
| :----------- | :-------: | :-------: | :-------: | 
| **nlpTurk**  | **97.78** | **96.43** | **97.10** |  
| **zemberek** |   77.48   |   95.20   |   85.43   | 


### NUM \[numeral\]

|              | precision | recall    | f1-score  | 
| :----------- | :-------: | :-------: | :-------: | 
| **nlpTurk**  | **95.54** | **98.02** | **96.76** |  
| **zemberek** |   89.89   |   89.62   |   89.76   | 


### PRON \[pronoun\]

|              | precision | recall    | f1-score  | 
| :----------- | :-------: | :-------: | :-------: | 
| **nlpTurk**  | **91.06** | **91.80** | **91.43** |  
| **zemberek** |   89.35   |   85.48   |   87.37   | 


### PROPN \[proper noun\]

|              | precision | recall    | f1-score  | 
| :----------- | :-------: | :-------: | :-------: | 
| **nlpTurk**  | **92.82** | **95.53** | **94.15** |  
| **zemberek** |   0.00    |   0.00    |   0.00    | 


### PUNCT \[punctuation\]

|              | precision | recall    | f1-score  | 
| :----------- | :-------: | :-------: | :-------: | 
| **nlpTurk**  | **98.72** | **99.37** | **99.04** |  
| **zemberek** |   97.61   |   99.05   |   98.32   | 


### SCONJ \[subordinating conjunction\]

|              | precision | recall    | f1-score  | 
| :----------- | :-------: | :-------: | :-------: | 
| **nlpTurk**  | **100.00**| **91.30** | **95.45** |  
| **zemberek** |   0.00    |   0.00    |   0.00    | 


### VERB \[verb\]

|              | precision | recall    | f1-score  | 
| :----------- | :-------: | :-------: | :-------: | 
| **nlpTurk**  | **94.60** | **97.65** | **96.10** |  
| **zemberek** |   89.48   |   96.24   |   92.74   | 


### X \[other\]

|              | precision | recall    | f1-score  | 
| :----------- | :-------: | :-------: | :-------: | 
| **nlpTurk**  | **29.31** | **44.74** | **35.42** |  
| **zemberek** |   0.00    |   0.00    |   0.00    | 
