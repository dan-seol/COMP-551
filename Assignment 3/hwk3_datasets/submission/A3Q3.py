
# coding: utf-8

# # Yelp Frequency Bag-of-Words

# In[1]:


# import essential libraries

import random
import string
import numpy as np
import pandas as pd
import operator as op
import scipy as sp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_sc, accuracy_sc #f1_sc(y_true, y_pred)

# ...
from sklearn.model_selection import GridSearchCV, PredefinedSplit, ParameterGrid
from sklearn.metrics import f1_sc
# for classifiers
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
import ast
from collections import Counter
import matplotlib.pyplot as plt


# In[2]:


# examples are split with  \n
# rating given with review is last char in example
yelp_tr = pd.read_csv("yelp-train.txt", sep='\t', lineterminator='\n', header=None, names=['review', 'label'])
yelp_te = pd.read_csv("yelp-test.txt", sep='\t', lineterminator='\n', header=None, names=['review', 'label'])
yelp_va = pd.read_csv("yelp-valid.txt", sep='\t', lineterminator='\n', header=None, names=['review', 'label'])
imdb_tr = pd.read_csv("IMDB-train.txt", sep='\t', lineterminator='\n', header=None, names=['review', 'label'])
imdb_te = pd.read_csv("IMDB-test.txt", sep='\t', lineterminator='\n', header=None, names=['review', 'label'])
imdb_va = pd.read_csv("IMDB-valid.txt", sep='\t', lineterminator='\n', header=None, names=['review', 'label'])



# In[3]:


# categories of given dataset
hw3_datasets = {
    'Yelp': {'train': yelp_tr, 'valid': yelp_va, 'test': yelp_te},
    'IMDB': {'train': imdb_tr, 'valid': imdb_va, 'test': imdb_te},
}


# In[4]:


#Pre-processing:
#You make the sentences to lower case

for dataset in hw3_datasets.values():
    for df in dataset.values():
        df['review'] = df['review'].str.lower()
        df['review'] = df['review'].str.replace('<br /><br />', ' ').str.replace('[^\w\s]', '')




# In[5]:


vocab = {}
#We exclude the words that do not have much semantic value: such as "the"
#NLTK's stop words list
stops = {'the','a','i','me', 'youre', 'not', 'my', 'myself','we','our','ours','ourselves','you','your','yours','yourself','yourselves','he','him','his','himself','she','her','hers','herself','it','its','itself','they','them','their','theirs','themselves','what','which','who','whom','this','that','these','those','am','is','are','was','were','be','been','being','have','has','had','having','do','does','did','doing','and','but','if','or','because','as','until','while','of','at','by','for','with','about','against','between','into','through','during','before','after','above','below','to','from','up','down','in','out','on','off','over','under','again','further','then','once', 'there','when','where','why','how','all','any','both','each','most','other','some','such','nor','only','so','than','too','very','s','t','can','will','just','don','should','now'}
for group_name, group in hw3_datasets.items():
    list_all_words = [word for sentence in group['train']['review'].str.split().tolist() for word in sentence]
    list_freq_words = Counter(word for word in list_all_words if word not in stops).most_common(10000)
    vocab[group_name] = {word[0]: i for i, word in enumerate(list_freq_words)}


# In[6]:


vtzrYelp = CountVectorizer(max_features = 10000, binary=False, vocabulary= vocab['Yelp']) #make it onehot encoded
train = hw3_datasets['Yelp']['train']
test = hw3_datasets['Yelp']['test']
val = hw3_datasets['Yelp']['valid']
train_vectors = vtzrYelp.fit_transform(train['review'])
test_vectors = vtzrYelp.transform(test['review'])
val_vectors = vtzrYelp.transform(val['review'])


# In[7]:


#print(train_vectors)
type(train_vectors) #what type does it return?
X = train_vectors.toarray() #change it to a 2d array
#print(sp.sparse.csr_matrix(train_vectors.toarray())) #could I change it back to sparse.csr_matrix?


# In[8]:


#Yelp_vocab_stored = pd.read_csv("/submission/Yelp-vocab.txt", sep='\t', lineterminator='\n', header=None, names=['word', 'index', 'count'])
#Yelp_counts = Yelp_vocab_stored['count'] # get the count column so that you can divide each entry of the vector
normalizer = Normalizer(norm='l1')

train_vectors_norm = normalizer.transform(train_vectors)
test_vectors_norm = normalizer.transform(test_vectors)
val_vectors_norm = normalizer.transform(val_vectors)


# In[9]:


bayes_params = ParameterGrid({}) #We have no hyperparameters to adjust..yet
tree_params = ParameterGrid({'random_state':[329],'criterion':['gini','entropy'],'max_depth':[None,10,100,1000],'min_samples_split':[2,5,10]})
svm_params = ParameterGrid({'random_state':[329],'loss':['hinge','squared_hinge'],'C':[1.0,.5,2.0,5.0]})

classifiers= [(GaussianNB, bayes_params), (DecisionTreeClassifier, tree_params), (svm.LinearSVC, svm_params)]


# In[10]:


def eval_csfier_quick(classifier):
    try:
        csfier.fit(train_vectors, train['label'])
        val_y = csfier.predict(val_vectors)
    except: #some of the classifiers can't deal with sparse matrices
        csfier.fit(train_vectors.toarray(), train['label'])
        val_y = csfier.predict(val_vectors.toarray())

    val_f1 = f1_sc(val['label'],val_y, average='macro')
    return val_f1

def test_classifier(classifier):
    try:
        csfier.fit(train_vectors, train['label'])
        test_y = csfier.predict(test_vectors)
    except: #some of the classifiers can't deal with sparse matrices
        csfier.fit(train_vectors.toarray(), train['label'])
        test_y = csfier.predict(test_vectors.toarray())

    test_f1 = f1_sc(test['label'],test_y, average='macro')
    return test_f1

def test_csfier_train(classifier):
    try:
        csfier.fit(train_vectors, train['label'])
        train_y = csfier.predict(train_vectors)
    except: #some of the classifiers can't deal with sparse matrices
        csfier.fit(train_vectors.toarray(), train['label'])
        train_y = csfier.predict(train_vectors.toarray())

    train_f1 = f1_sc(train['label'], train_y, average='macro')
    return train_f1


# In[11]:


# find best params for a classifier
def optimal_parameters(classifier, param_grid):
    opt_sc=0 #f1 sc on validation
    best_params=None
    for params in param_grid:
        print(f"Trying: {params}")
        sc = eval_csfier_quick(classifier(**params))
        print(f"F1 sc Validation: {sc}\n")
        if sc>opt_sc:
            opt_sc=sc
            best_params=params

    print(f"Best params for Validation: {best_params}")
    print(f"Best F1 sc on Validation: {opt_sc}\n")

    return classifier(**best_params)


# In[31]:


def optimal_GaussianNB(vec_smoothing):
    opt_sc = 0
    best_params = -1
    for i in range(len(vec_smoothing)):
        print(f"Trying:{vec_smoothing[i]}")
        csfier = GaussianNB(priors=None, var_smoothing=vec_smoothing[i])
        csfier.fit(train_vectors.toarray(), train['label'])
        val_y = csfier.predict(val_vectors.toarray())
        val_f1 = f1_sc(val['label'],val_y, average='macro')
        print(f"F1 sc Validation: {val_f1}\n")
        if  val_f1>opt_sc:
            opt_sc= val_f1
            best_params=vec_smoothing[i]
    print(f"Best params for Validation: {best_params}")
    print(f"Best F1 sc on Validation: {opt_sc}\n")
    return best_params


# In[13]:


for pair in classifiers: # cycle through the classifiers and parameters
    classifier = pair[0]
    param_grid = pair[1]
    print(classifier)
    best_classifier = optimal_parameters(classifier,param_grid)
    print(f"Test sc for best params: {test_classifier(best_classifier)}\n")
    print(f"Train sc for best params: {test_csfier_train(best_classifier)}\n")


# In[34]:


#tuning Var_smoothing for GaussianNB
vec_smt = [1e-5,1e-6,1e-7, 1e-8, 1e-9, 1e-10, 1e-11,1e-12]
best_GNB = optimal_GaussianNB(vec_smt)
GNB = GaussianNB(priors=None, var_smoothing = best_GNB)
GNB.fit(train_vectors.toarray(), train['label'])
GNB_hat = GNB.predict(test_vectors.toarray())
GNB_f1_te = f1_sc(test['label'],GNB_hat, average='macro')
GNB_hat_tr = GNB.predict(train_vectors.toarray())
GNB_f1_tr = f1_sc(train['label'],GNB_hat_tr, average='macro')
print(f"Test sc for best params: {GNB_f1_te}\n")
print(f"Train sc for best params: {GNB_f1_tr}\n")


# In[30]:
