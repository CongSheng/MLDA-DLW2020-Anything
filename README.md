# MLDA-DLW2020-Anything

This is the code repo for team Anything's submission to MLDA Deep Learning Week 2020 Hackathon.

Team members:
- Thien Tran ([gau-nernst](https://github.com/gau-nernst))
- Leow Cong Sheng ([CongSheng](https://github.com/CongSheng))

## Setup

The key library used is [FastText](https://fasttext.cc/). There are also other common libraries such as `numpy`, `pandas` and `matplotlib`.

## Datasets

Three datasets are used:
- Resume dataset
    - https://www.kaggle.com/avanisiddhapura27/resume-dataset
    - https://www.kaggle.com/oo7kartik/resume-text-batch?
- JD dataset
    - https://www.kaggle.com/bman93/dataset

A copy of each dataset is also included in this repo, zipped inside `datasets.zip`.

## Explanation

[Clean data.ipynb](Clean%20data.ipynb) imports the data and does common data cleaning techniques, such as removing stopwords, strange characters. It writes `data.txt` and `jd.csv` to disc for training word embeddings with FastText later.

With FastText, train word embeddings on `data.txt` (no label, unsupervised) and on `jd.csv` (with labels, supervised).

```python
import fasttext
from tf_projector import export_to_tf_projector

model = fasttext.train_unsupervised('data.txt')
export_to_tf_projector(model, 'anything')
model.save_model('embeddings.bin')

model = fasttext.train_supervised('jd.csv')
export_to_tf_projector(model, 'jd')
model.save_model('jd.bin')
```

`export_to_tf_projector()` is a helper function to export word vectors and metadata to [TensorFlow Projector](https://projector.tensorflow.org/) compatible format for visualization.

Our results show that there is no interesting insights from the model trained with `jd.csv`.

[fasttext.ipynb](fasttext.ipynb) demonstrates two simple ways to use the word embeddings
1. Use `model.get_nearest_neighbors()` to get similar skills and job titles to a given skill or job
2. Use `model.get_sentence_vector()` to generate vector representation of job descriptions. The vectors show that different job titles belong to distinct clusters, implying the potential to map skillsets with job requirements.
