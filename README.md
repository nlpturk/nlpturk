# nlpTurk - Türkçe Doğal Dil İşleme Kütüphanesi

nlpTurk, makine öğrenmesi tabanlı cümle sınırı belirleme, kelime kökü bulma (lemmatization), metin parçası etiketleme (POS tagging) modellerinden oluşan açık kaynak Türkçe Doğal Dil işleme kütüphanesidir.

## Kurulum ve Kullanım

nlpTurk kütüphanesini [PyPI](https://pypi.org/project/nlpturk/)'den indirerek kullanmaya başlayabilirsiniz. 
 
```bash
pip install nlpturk
```

```python
import nlpturk

metin = "Sosyal medya hayatımıza hızlı girdi.ama yazım kurallarına dikkat eden pek yok :)"
doc = nlpturk(metin)

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
    print(f"cümle #{i+1}: {sent.text}")
    for token in sent:
        print(f"  token: {token.text}, lemma: {token.lemma}, pos: {token.pos}")

"""
Prints:
  cümle #1: Sosyal medya hayatımıza hızlı girdi.
    token: Sosyal, lemma: sosyal, pos: ADJ
    ...
  cümle #2: ama yazım kurallarına dikkat eden pek yok :)
    token: ama, lemma: ama, pos: CCONJ
    ...
"""
```

## Performans

Test veri seti değerlendirme sonuçları aşağıda yer almaktadır. Detaylı performans ve kıyaslama tablolarına [buradan](https://github.com/nlpturk/nlpturk/blob/master/benchmarks) ulaşabilirsiniz. 

|                                               | accuracy | precision | recall | f1-score | 
| :-------------------------------------------- | :------: | :-------: | :----: | :------: | 
| **SBD**<br/>(Cümle Sınırı Belirleme)          |    -     |   98.09   |  96.05 |  97.06   |  
| **POS Tagger**<br/>(Metin Parçası Etiketleme) |    -     |   95.75   |  96.26 |  96.01   |   
| **Lemmatizer**<br/>(Kelime Kökü Bulma)        |  96.87   |     -     |    -   |    -     |

<br/>Kendi veri setiniz üzerinden performans değerlendirmesi yapmak için;

```bash
git clone https://github.com/nlpturk/nlpturk.git
cd nlpturk
pip install -r requirements.txt
python -m nlpturk benchmark --data_path path/to/data --output_path path/to/output
```

## Websitesi

nlpTurk kütüphanesini kurulum yapmadan deneyimlemek için demo [websitesine](http://18.133.122.155) göz atabilirsiniz.

![home](https://github.com/nlpturk/nlpturk/blob/master/website/home.png)

## Kaynaklar

nlpTurk kütüphanesi modelleri geliştirilirken aşağıda listelenen veri setlerinden yararlanılmıştır. Veri setleri toplamda yaklaşık 64k cümle ve 650k token'dan oluşmaktadır.

**UD Turkish Atis v2.10**

Yazar: Köse, Mehmet; Yıldız, Olcay Taner
<br/>URL: https://github.com/UniversalDependencies/UD_Turkish-Atis
<br/>Lisans: CC BY-SA 4.0
<br/>5432 cümle, 45611 token

**UD Turkish FrameNet v2.10**

Yazar: Cesur, Neslihan; Kuzgun, Aslı; Yıldız, Olcay Taner; Marşan, Büşra; Kuyrukçu, Oğuzhan; Arıcan, Bilge Nas; Sanıyar, Ezgi; Kara, Neslihan; Özçelik, Merve
<br/>URL: https://github.com/UniversalDependencies/UD_Turkish-FrameNet
<br/>Lisans: CC BY-SA 4.0
<br/>2698 cümle, 17637 token

**UD Turkish Kenet v2.10**

Yazar: Kuzgun, Aslı; Cesur, Neslihan; Yıldız, Olcay Taner; Kuyrukçu, Oğuzhan; Yenice, Arife Betül; Arıcan, Bilge Nas; Sanıyar, Ezgi
<br/>URL: https://github.com/UniversalDependencies/UD_Turkish-Kenet
<br/>Lisans: CC BY-SA 4.0
<br/>18687 cümle, 168230 token

**UD Turkish Penn v2.10**

Yazar: Cesur, Neslihan; Kuzgun, Aslı; Yıldız, Olcay Taner; Marşan, Büşra; Kara, Neslihan; Arıcan, Bilge Nas; Özçelik, Merve; Aslan, Deniz Baran
<br/>URL: https://github.com/UniversalDependencies/UD_Turkish-Penn
<br/>Lisans: CC BY-SA 4.0
<br/>16396 cümle, 173957 token

**trseg-41 v1.0**

Yazar: Kurtuluş, Emirhan; Safaya, Ali; Goktogan, Arda
<br/>URL: https://data.tdd.ai/#/72207c43-e123-4ce9-aa8a-84af68181e47
<br/>Lisans: CC BY 4.0
<br/>20616 cümle, 248125 token