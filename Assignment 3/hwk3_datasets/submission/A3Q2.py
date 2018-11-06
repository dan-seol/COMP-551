
# coding: utf-8

# # Yelp Binary Bag-of-Words

# In[17]:


# import essential libraries

import random
import string
import numpy as np
import pandas as pd
import operator as op
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, accuracy_score #f1_score(y_true, y_pred)

# ...
from sklearn.model_selection import GridSearchCV, PredefinedSplit, ParameterGrid
from sklearn.metrics import f1_score
# for csfiers
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
import ast
from collections import Counter
from tqdm import tqdm


# In[18]:


# examples are split with  \n
# rating given with review is last char in example
yelp_tr = pd.read_csv("yelp-train.txt", sep='\t', lineterminator='\n', header=None, names=['review', 'label'])
yelp_te = pd.read_csv("yelp-test.txt", sep='\t', lineterminator='\n', header=None, names=['review', 'label'])
yelp_va = pd.read_csv("yelp-valid.txt", sep='\t', lineterminator='\n', header=None, names=['review', 'label'])
imdb_tr = pd.read_csv("IMDB-train.txt", sep='\t', lineterminator='\n', header=None, names=['review', 'label'])
imdb_te = pd.read_csv("IMDB-test.txt", sep='\t', lineterminator='\n', header=None, names=['review', 'label'])
imdb_va = pd.read_csv("IMDB-valid.txt", sep='\t', lineterminator='\n', header=None, names=['review', 'label'])



# In[19]:


# categories of given dataset
hw3_datasets = {
    'Yelp': {'train': yelp_tr, 'valid': yelp_va, 'test': yelp_te},
    'IMDB': {'train': imdb_tr, 'valid': imdb_va, 'test': imdb_te},
}


# In[20]:


#Pre-processing:
#You make the sentences to lower case

for dataset in hw3_datasets.values():
    for df in dataset.values():
        df['review'] = df['review'].str.lower()
        df['review'] = df['review'].str.replace('<br /><br />', ' ').str.replace('[^\w\s]', '')




# In[21]:


vocab = {}
#We exclude the words that do not have much semantic value: such as "the"
#NLTK's stop words list
stops = {'the','a','i','me', 'youre', 'not', 'my', 'myself','we','our','ours','ourselves','you','your','yours','yourself','yourselves','he','him','his','himself','she','her','hers','herself','it','its','itself','they','them','their','theirs','themselves','what','which','who','whom','this','that','these','those','am','is','are','was','were','be','been','being','have','has','had','having','do','does','did','doing','and','but','if','or','because','as','until','while','of','at','by','for','with','about','against','between','into','through','during','before','after','above','below','to','from','up','down','in','out','on','off','over','under','again','further','then','once', 'there','when','where','why','how','all','any','both','each','most','other','some','such','nor','only','so','than','too','very','s','t','can','will','just','don','should','now'}
for group_name, group in hw3_datasets.items():
    list_all_words = [word for sentence in group['train']['review'].str.split().tolist() for word in sentence]
    list_freq_words = Counter(word for word in list_all_words if word not in stops).most_common(10000)
    vocab[group_name] = {word[0]: i for i, word in enumerate(list_freq_words)}


# In[22]:


vtzrYelp = CountVectorizer(max_features = 10000, binary=True, vocabulary= vocab['Yelp']) #make it onehot encoded
train = hw3_datasets['Yelp']['train']
test = hw3_datasets['Yelp']['test']
val = hw3_datasets['Yelp']['valid']
train_vectors = vtzrYelp.fit_transform(train['review'])
test_vectors = vtzrYelp.transform(test['review'])
val_vectors = vtzrYelp.transform(val['review'])


# In[23]:


def eval_csfier(csfier):

    try:
        csfier.fit(train_vectors, train['label'])
        train_y =csfier.predict(train_vectors)
        test_y = csfier.predict(test_vectors)
        val_y = csfier.predict(val_vectors)

    except: #some of the csfiers can't deal with sparse matrices
        csfier.fit(train_vectors.toarray(), train['label'])
        train_y =csfier.predict(train_vectors.toarray())
        test_y = csfier.predict(test_vectors.toarray())
        val_y = csfier.predict(val_vectors.toarray())

    train_score = accuracy_score(train['label'],train_y)
    val_score = accuracy_score(val['label'],val_y)
    test_score = accuracy_score(test['label'],test_y)

    train_f1 = f1_score(train['label'],train_y, average='macro')
    val_f1 = f1_score(val['label'],val_y, average='macro')
    test_f1 = f1_score(test['label'],test_y, average='macro')

    print(type(csfier))
    print(f"Train Acc: {train_score}")
    print(f"Val Acc: {val_score}")
    print(f"Test Acc: {test_score}")
    print("\n")
    print(f"Train F1: {train_f1}")
    print(f"Val F1: {val_f1}")
    print(f"Test F1: {test_f1}")
    print("\n")
    return val_f1


# In[8]:


def eval_csfier_quick(csfier):
    try:
        csfier.fit(train_vectors, train['label'])
        val_y = csfier.predict(val_vectors)
    except: #some of the csfiers can't deal with sparse matrices
        csfier.fit(train_vectors.toarray(), train['label'])
        val_y = csfier.predict(val_vectors.toarray())

    val_f1 = f1_score(val['label'],val_y, average='macro')
    return val_f1

def test_csfier(csfier):
    try:
        csfier.fit(train_vectors, train['label'])
        test_y = csfier.predict(test_vectors)
    except: #some of the csfiers can't deal with sparse matrices
        csfier.fit(train_vectors.toarray(), train['label'])
        test_y = csfier.predict(test_vectors.toarray())

    test_f1 = f1_score(test['label'],test_y, average='macro')
    return test_f1

def test_csfier_train(csfier):
    try:
        csfier.fit(train_vectors, train['label'])
        train_y = csfier.predict(train_vectors)
    except: #some of the csfiers can't deal with sparse matrices
        csfier.fit(train_vectors.toarray(), train['label'])
        train_y = csfier.predict(train_vectors.toarray())

    train_f1 = f1_score(train['label'], train_y, average='macro')
    return train_f1


# In[24]:


random = DummyClassifier(strategy='uniform', random_state=329) #set random seed so we get consistent results
maj = DummyClassifier(strategy='most_frequent')

print(f"F1 Score of Random csfier on Test: {test_csfier(random)}\n")
print(f"F1 Score of Majority csfier on Test: {test_csfier(maj)}")


# In[25]:


bayes_params = ParameterGrid({'alpha':[.1,.5,1,2]})
tree_params = ParameterGrid({'random_state':[329],'criterion':['gini','entropy'],'max_depth':[None,10,100,1000],'min_samples_split':[2,5,10]})
svm_params = ParameterGrid({'random_state':[329],'loss':['hinge','squared_hinge'],'C':[1.0,.5,2.0,5.0]})

csfiers= [(BernoulliNB, bayes_params), (DecisionTreeClassifier, tree_params), (svm.LinearSVC, svm_params)]



# In[26]:


# find best params for a csfier
def optimize_parameters(csfier, param_grid):
    best_score=0 #f1 score on validation
    best_params=None
    for params in param_grid:
        print(f"Trying: {params}")
        score = eval_csfier_quick(csfier(**params))
        print(f"F1 Score Validation: {score}\n")
        if score>best_score:
            best_score=score
            best_params=params

    print(f"Best params for Validation: {best_params}")
    print(f"Best F1 Score on Validation: {best_score}\n")

    return csfier(**best_params)


# In[16]:



for pair in csfiers: # cycle through the csfiers and parameters
    csfier = pair[0]
    param_grid = pair[1]
    print(csfier)
    best_csfier = optimize_parameters(csfier,param_grid)
    print(f"Test score for best params: {test_csfier(best_csfier)}\n")
    print(f"Train score for best params: {test_csfier_train(best_csfier)}\n")
