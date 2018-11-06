# import essential libraries

import random
import string
import numpy as np
import pandas as pd
import operator as op
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
# ...
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import f1_score
# for classifiers
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
import ast
from collections import Counter
from tqdm import tqdm

yelp_tr = pd.read_csv("hwk3_datasets/yelp-train.txt", sep='\t', lineterminator='\n', header=None, names=['review', 'label'])
yelp_te = pd.read_csv("hwk3_datasets/yelp-test.txt", sep='\t', lineterminator='\n', header=None, names=['review', 'label'])
yelp_va = pd.read_csv("hwk3_datasets/yelp-valid.txt", sep='\t', lineterminator='\n', header=None, names=['review', 'label'])
imdb_tr = pd.read_csv("hwk3_datasets/IMDB-train.txt", sep='\t', lineterminator='\n', header=None, names=['review', 'label'])
imdb_te = pd.read_csv("hwk3_datasets/IMDB-test.txt", sep='\t', lineterminator='\n', header=None, names=['review', 'label'])
imdb_va = pd.read_csv("hwk3_datasets/IMDB-valid.txt", sep='\t', lineterminator='\n', header=None, names=['review', 'label'])
# categories of given dataset
hw3_datasets = {
    'Yelp': {'train': yelp_tr, 'valid': yelp_va, 'test': yelp_te},
    'IMDB': {'train': imdb_tr, 'valid': imdb_va, 'test': imdb_te},
}
#Pre-processing:
#You make the sentences to lower case

for dataset in hw3_datasets.values():
    for df in dataset.values():
        df['review'] = df['review'].str.lower()
        df['review'] = df['review'].str.replace('<br /><br />', ' ').str.replace('[^\w\s]', '')

vocab = {}
#We exclude the words that do not have much semantic value: such as "the"
#NLTK's stop words list
stops = {'the','a','i','me', 'youre', 'not', 'my', 'myself','we','our','ours','ourselves','you','your','yours','yourself','yourselves','he','him','his','himself','she','her','hers','herself','it','its','itself','they','them','their','theirs','themselves','what','which','who','whom','this','that','these','those','am','is','are','was','were','be','been','being','have','has','had','having','do','does','did','doing','and','but','if','or','because','as','until','while','of','at','by','for','with','about','against','between','into','through','during','before','after','above','below','to','from','up','down','in','out','on','off','over','under','again','further','then','once', 'there','when','where','why','how','all','any','both','each','most','other','some','such','nor','only','so','than','too','very','s','t','can','will','just','don','should','now'}
for group_name, group in hw3_datasets.items():
    list_all_words = [word for sentence in group['train']['review'].str.split().tolist() for word in sentence]
    list_freq_words = Counter(word for word in list_all_words if word not in stops).most_common(10000)
    vocab[group_name] = {word[0]: i for i, word in enumerate(list_freq_words)}
vtzrIMDB = CountVectorizer(max_features = 10000, binary=True, vocabulary= vocab['IMDB']) #make it onehot encoded
train = hw3_datasets['IMDB']['train']
test = hw3_datasets['IMDB']['test']
val = hw3_datasets['IMDB']['valid']
train_vectors = vtzrIMDB.fit_transform(train['review'])
test_vectors = vtzrIMDB.transform(test['review'])
val_vectors = vtzrIMDB.transform(val['review'])
