from thinc.api import Config
from spacy.util import load_config_from_str


_default_t2v_configs = '''
[paths]
train = null
dev = null
vectors = null
init_tok2vec = null
source = null
 
[system]
gpu_allocator = null
seed = 0

[nlp]
lang = "tr"
pipeline = ["tok2vec","sbd","tagger","lemmatizer"]
batch_size = 1000
disabled = []
before_creation = null
after_creation = null
after_pipeline_creation = null
tokenizer = {"@tokenizers":"spacy.Tokenizer.v1"}

[components]

[components.sbd]
factory = "sbd"

[components.sbd.model]
@architectures = "spacy.Tagger.v2"
nO = null
normalize = false

[components.sbd.model.tok2vec]
@architectures = "spacy.Tok2VecListener.v1"
width = ${components.tok2vec.model.encode.width}
upstream = "*"

[components.tagger]
factory = "tagger"
neg_prefix = "!"
overwrite = false
scorer = {"@scorers":"spacy.tagger_scorer.v1"}

[components.tagger.model]
@architectures = "spacy.Tagger.v2"
nO = null
normalize = false

[components.tagger.model.tok2vec]
@architectures = "spacy.Tok2VecListener.v1"
width = ${components.tok2vec.model.encode.width}
upstream = "*"

[components.lemmatizer]
factory = "trainable_lemmatizer"
backoff = "orth"
min_tree_freq = 1
overwrite = false
scorer = {"@scorers":"spacy.lemmatizer_scorer.v1"}
top_k = 5

[components.lemmatizer.model]
@architectures = "spacy.Tagger.v2"
nO = null
normalize = false

[components.lemmatizer.model.tok2vec]
@architectures = "spacy.Tok2VecListener.v1"
width = ${components.tok2vec.model.encode.width}
upstream = "*"

[components.tok2vec]
factory = "tok2vec"

[components.tok2vec.model]
@architectures = "spacy.Tok2Vec.v2"

[components.tok2vec.model.embed]
@architectures = "spacy.MultiHashEmbed.v2"
width = ${components.tok2vec.model.encode.width}
attrs = ["NORM","PREFIX","SUFFIX","SHAPE"]
rows = [5000,2500,2500,2500]
include_static_vectors = false

[components.tok2vec.model.encode]
@architectures = "spacy.MaxoutWindowEncoder.v2"
width = 300
depth = 8
window_size = 1
maxout_pieces = 3

[corpora]

[corpora.dev]
@readers = "spacy.Corpus.v1"
path = ${paths.dev}
max_length = 0
gold_preproc = false
limit = 0
augmenter = null

[corpora.train]
@readers = "spacy.Corpus.v1"
path = ${paths.train}
max_length = 0
gold_preproc = false
limit = 0
augmenter = null

[training]
dev_corpus = "corpora.dev"
train_corpus = "corpora.train"
seed = ${system.seed}
gpu_allocator = ${system.gpu_allocator}
dropout = 0.1
accumulate_gradient = 1
patience = 0
max_epochs = 0
max_steps = 20000
eval_frequency = 200
frozen_components = []
annotating_components = []
before_to_disk = null

[training.batcher]
@batchers = "spacy.batch_by_words.v1"
discard_oversize = false
tolerance = 0.2
get_length = null

[training.batcher.size]
@schedules = "compounding.v1"
start = 100
stop = 1000
compound = 1.001
t = 0.0

[training.logger]
@loggers = "spacy.ConsoleLogger.v1"
progress_bar = false

[training.optimizer]
@optimizers = "Adam.v1"
beta1 = 0.9
beta2 = 0.999
L2_is_weight_decay = true
L2 = 0.01
grad_clip = 1.0
use_averages = false
eps = 0.00000001
learn_rate = 0.001

[training.score_weights]
sbd_f = 0.33
sbd_p = null
sbd_r = null
tag_acc = 0.33
lemma_acc = 0.33

[pretraining]

[initialize]
vectors = ${paths.vectors}
init_tok2vec = ${paths.init_tok2vec}
vocab_data = null
lookups = null
before_init = null
after_init = null

[initialize.components]

[initialize.tokenizer]
'''

_default_trf_configs = '''
[paths]
train = null
dev = null
vectors = null
init_tok2vec = null
source = null

[system]
gpu_allocator = "pytorch"
seed = 0

[nlp]
lang = "tr"
pipeline = ["transformer","sbd","tagger","lemmatizer"]
batch_size = 128
disabled = []
before_creation = null
after_creation = null
after_pipeline_creation = null
tokenizer = {"@tokenizers":"spacy.Tokenizer.v1"}

[components]

[components.sbd]
factory = "sbd"

[components.sbd.model]
@architectures = "spacy.Tagger.v2"
nO = null
normalize = false

[components.sbd.model.tok2vec]
@architectures = "spacy-transformers.TransformerListener.v1"
grad_factor = 1.0
pooling = {"@layers":"reduce_mean.v1"}
upstream = "*"

[components.tagger]
factory = "tagger"
neg_prefix = "!"
overwrite = false
scorer = {"@scorers":"spacy.tagger_scorer.v1"}

[components.tagger.model]
@architectures = "spacy.Tagger.v2"
nO = null
normalize = false

[components.tagger.model.tok2vec]
@architectures = "spacy-transformers.TransformerListener.v1"
grad_factor = 1.0
pooling = {"@layers":"reduce_mean.v1"}
upstream = "*"

[components.lemmatizer]
factory = "trainable_lemmatizer"
backoff = "orth"
min_tree_freq = 1
overwrite = false
scorer = {"@scorers":"spacy.lemmatizer_scorer.v1"}
top_k = 5

[components.lemmatizer.model]
@architectures = "spacy.Tagger.v2"
nO = null
normalize = false

[components.lemmatizer.model.tok2vec]
@architectures = "spacy-transformers.TransformerListener.v1"
grad_factor = 1.0
pooling = {"@layers":"reduce_mean.v1"}
upstream = "*"

[components.transformer]
factory = "transformer"
max_batch_items = 4096
set_extra_annotations = {"@annotation_setters":"spacy-transformers.null_annotation_setter.v1"}

[components.transformer.model]
@architectures = "spacy-transformers.TransformerModel.v3"
name = null
mixed_precision = false

[components.transformer.model.get_spans]
@span_getters = "spacy-transformers.strided_spans.v1"
window = 128
stride = 96

[components.transformer.model.grad_scaler_config]

[components.transformer.model.tokenizer_config]
use_fast = true

[components.transformer.model.transformer_config]

[corpora]

[corpora.dev]
@readers = "spacy.Corpus.v1"
path = ${paths.dev}
max_length = 0
gold_preproc = false
limit = 0
augmenter = null

[corpora.train]
@readers = "spacy.Corpus.v1"
path = ${paths.train}
max_length = 0
gold_preproc = false
limit = 0
augmenter = null

[training]
accumulate_gradient = 3
dev_corpus = "corpora.dev"
train_corpus = "corpora.train"
seed = ${system.seed}
gpu_allocator = ${system.gpu_allocator}
dropout = 0.1
patience = 0
max_epochs = 0
max_steps = 20000
eval_frequency = 200
frozen_components = []
annotating_components = []
before_to_disk = null

[training.batcher]
@batchers = "spacy.batch_by_padded.v1"
discard_oversize = true
size = 2000
buffer = 256
get_length = null

[training.logger]
@loggers = "spacy.ConsoleLogger.v1"
progress_bar = false

[training.optimizer]
@optimizers = "Adam.v1"
beta1 = 0.9
beta2 = 0.999
L2_is_weight_decay = true
L2 = 0.01
grad_clip = 1.0
use_averages = false
eps = 0.00000001

[training.optimizer.learn_rate]
@schedules = "warmup_linear.v1"
warmup_steps = 250
total_steps = 20000
initial_rate = 0.00005

[training.score_weights]
sbd_f = 0.33
sbd_p = null
sbd_r = null
tag_acc = 0.33
lemma_acc = 0.33

[pretraining]

[initialize]
vectors = ${paths.vectors}
init_tok2vec = ${paths.init_tok2vec}
vocab_data = null
lookups = null
before_init = null
after_init = null

[initialize.components]

[initialize.tokenizer]
'''


def load_default_configs(trf_model: str = None) -> Config:
    """Load default configs.

    Args:
        trf_model (str, optional): Transformer model name. Defaults to None.

    Returns:
        Config: The loaded configs.
    """
    if trf_model:
        configs = load_config_from_str(_default_trf_configs)
        configs['components']['transformer']['model']['name'] = trf_model
        return configs
    else:
        return load_config_from_str(_default_t2v_configs)
