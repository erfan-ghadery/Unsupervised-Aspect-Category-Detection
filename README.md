# Unsupervised-Aspect-Category-Detection
This repository contains the code for the paper "An Unsupervised Approach for Aspect Category Detection Using Soft Cosine Similarity Measure".

## Data
The unlabeld yelp reviews sentences can be downloaded at [[Download]](https://drive.google.com/file/d/1aCOK59-hWj9qmFT7jsYb4N791Ty9tvNx/view). Put this file in the 'yelp-weak-supervision' folder.
The pre-trained word embeddings can be downloaded at [[Download]](https://drive.google.com/file/d/1Uh7TOEqthjbzIUHIOQ2EYH1nLzVhpLrn/view). Put this file in the 'word-embedding' folder.

## Dependencies

* python 3.6.0
* numpy 1.15.4
* gensim 3.4.0
* tqdm

## Dataset

You can find the dataset in the semeval 2014 website [here](http://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools). Copy the dataset in the directory 'dataset'

## Cite
If you use the code, please cite the following paper:
```
@article{ghadery2018unsupervised,
  title={An Unsupervised Approach for Aspect Category Detection Using Soft Cosine Similarity Measure},
  author={Ghadery, Erfan and Movahedi, Sajad and Faili, Heshaam and Shakery, Azadeh},
  journal={arXiv preprint arXiv:1812.03361},
  year={2018}
}
```
