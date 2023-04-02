This repository is the fork of the official implementation of the paper ["F-COREF: Fast, Accurate and Easy to Use Coreference Resolution"](https://arxiv.org/abs/2209.04280). For all possible usages of the `fastcoref` package please refer to [the official repository of the paper](https://github.com/shon-otmazgin/fastcoref). This repository shows examples on how to use the `fastcoref` library for the processing of Ukrainian documents by the corresponding pre-trained model.

The `fastcoref` Python package provides an easy and fast API for coreference information with only few lines of code without any prepossessing steps.

- [Installation](#installation)
- [Demo](#demo)
- [Quick start](#quick-start)
- [Citation](#citation)

## Installation

```python
pip install fastcoref
```

## Demo

[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1vsaH15DFDrmKB4aNsQ-9TCQGTW73uk1y?usp=sharing)

## Quick start

The main functionally of the package is the `predict` function.
The return value of the function is a list of `CorefResult` objects, from which one can extract the coreference clusters (either as strings or as character indices over the original texts), as well as the logits for each corefering entity pair:

```python
from fastcoref import FCoref
import spacy

nlp = spacy.load('uk_core_news_md')

model_path = "artemkramov/coref-ua"
model = FCoref(model_name_or_path=model_path, device='cuda:0', nlp=nlp)

preds = model.predict(
   texts=["""Мій друг дав мені свою машину та ключі до неї; крім того, він дав мені його книгу. Я з радістю її читаю."""]
)

preds[0].get_clusters(as_strings=False)
> [[(0, 3), (13, 17), (66, 70), (83, 84)],
 [(0, 8), (18, 22), (58, 61), (71, 75)],
 [(18, 29), (42, 45)],
 [(71, 81), (95, 97)]]

preds[0].get_clusters()
> [['Мій', 'мені', 'мені', 'Я'], ['Мій друг', 'свою', 'він', 'його'], ['свою машину', 'неї'], ['його книгу', 'її']]

preds[0].get_logit(
   span_i=(13, 17), span_j=(42, 45)
)

> -6.867196
```

if your text is already tokenized use `is_split_into_words=True`
```python
preds = model.predict(
   texts = [["Мій", "друг", "дав", "мені", "свою", "машину", "."]],
   is_split_into_words=True
)
```

Processing can be applied to a collection of texts of any length in a batched and parallel fashion:

```python
texts = ['text 1', 'text 2',.., 'text n']

# control the batch size 
# with max_tokens_in_batch parameter

preds = model.predict(
    texts=texts, max_tokens_in_batch=100
)
```

The `max_tokens_in_batch` parameter can be used to control the speed vs. memory consumption (as well as speed vs. accuracy) tradeoff, and can be tuned to maximize the utilization of the associated hardware.

## Citation

```
@inproceedings{Otmazgin2022FcorefFA,
  title={F-coref: Fast, Accurate and Easy to Use Coreference Resolution},
  author={Shon Otmazgin and Arie Cattan and Yoav Goldberg},
  booktitle={AACL},
  year={2022}
}
```

[F-coref: Fast, Accurate and Easy to Use Coreference Resolution](https://aclanthology.org/2022.aacl-demo.6) (Otmazgin et al., AACL-IJCNLP 2022)
