# -*- coding: utf-8 -*-
"""
Created on Sun May  3 12:21:14 2020

@author: Anang Kunaefi

description:
    text classification t1 (task1), t2 (task2), t3 (task3)
"""

import pandas as pd
import numpy as np
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.parse.stanford import StanfordParser
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn import model_selection, naive_bayes, svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import re
import string
import os
import pickle
from pprint import pprint
from custom_vectorizer import CustomVectorizer, PunctVectorizer
from datetime import datetime
# from contractions import contraction


'''
Supporting Class and functions for features union
'''
class TextSelector(BaseEstimator, TransformerMixin):
    def __init__(self, field):
        self.field = field
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.field]

class NumberSelector(BaseEstimator, TransformerMixin):
    def __init__(self, field):
        self.field = field
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[[self.field]]

def print_stats(preds, target, labels, sep='-', sep_len=40, fig_size=(10,8)):
    print('Accuracy = %.3f' % metrics.accuracy_score(target, preds))
    print(sep*sep_len)
    print('Classification report:')
    print(metrics.classification_report(target, preds))
    print(sep*sep_len)
    print('Confusion matrix')
    cm=metrics.confusion_matrix(target, preds)
    cm = cm / np.sum(cm, axis=1)[:,None]
    sns.set(rc={'figure.figsize':fig_size})
    sns.heatmap(cm,
        xticklabels=labels,
        yticklabels=labels,
           annot=True, cmap = 'YlGnBu')
    plt.pause(10)

def get_dataset(task):
    
    # get the dataset after text preprocessing
    
    # get feature after text preprocessing
        
    if task == 1:
        df_feat = pd.read_pickle('../data/feature_extraction_task1.pkl')
        df = df_feat.filter(['lemmatized_review','category',
                        'total_words','punctuation','num_np','num_vp',
                        'num_md','grammar','indicator'],axis=1)
        
    elif task == 2:
        df_feat = pd.read_pickle('../data/feature_extraction_task2.pkl')
        df = df_feat.filter(['lemmatized_review','category',
                        'total_words','punctuation','num_np','num_vp',
                        'num_md','grammar','indicator'],axis=1)
        invalid_columns = df[df.category > 3].index
        df.drop(invalid_columns,inplace=True)
        
    elif task == 3:       
        df_feat = pd.read_pickle('../data/feature_extraction_task3.pkl')
        df = df_feat.filter(['lemmatized_review','decision_category',
                        'total_words','punctuation','num_np','num_vp',
                        'num_md','grammar','indicator'],axis=1)

    
    # jaga2 jika ada null values
    null_index = df[df['lemmatized_review'] == ""].index 
    df.drop(null_index,inplace=True) 
    
  
    null_columns=df.columns[df.isnull().any()]
    # df.drop(null_columns,inplace=True)
    df.dropna(how='any',inplace=True)
    print(df[null_columns].isnull().sum())
    
    return df



# END of class and function

# SETTING
TASK = 3
start = datetime.now()

'''
# 1. The corpus
# The corpus is already processed in feature_extraction,
# and saved in the format of pandas dataframe, then pickled
# column: lemmatized_review, total_words, num_of_np, num_of_vp, num_of_md
'''
print('getting the data...')
df_reviews = get_dataset(TASK)
print(df_reviews.columns.tolist())

# Prepare train and test data#
if TASK == 1:
    combined_features = ['lemmatized_review','total_words','punctuation', 'num_np',
                         'num_vp','num_md','grammar','indicator']
    column_label = 'category'
elif TASK == 2:
    combined_features = ['lemmatized_review','total_words','punctuation', 'num_np',
                         'num_vp','num_md','grammar','indicator']
    column_label = 'category'   
elif TASK == 3:
    combined_features = ['lemmatized_review','total_words',
                         'punctuation','num_np','num_vp',
                         'num_md','grammar','indicator']
    column_label = 'decision_category'   

Train_X, Test_X, Train_Y,  Test_Y = model_selection.train_test_split(df_reviews[combined_features],df_reviews[column_label],test_size=0.2, random_state= 0)

'''
# Step 2. Use ML Algorithm Using Pipeline
# Gabungkan feature
'''

# all features combined
all_feats = FeatureUnion([
        ('text2', Pipeline([
            ('colext', TextSelector('lemmatized_review')),
            ('tfidf', TfidfVectorizer(min_df=.0025, max_df=0.25, ngram_range=(1,3)))
        ])),
        ('text', Pipeline([
            ('colext', TextSelector('lemmatized_review')),
            ('vect', CustomVectorizer()),
            ('tfidf', TfidfTransformer()),
        ])),
        ('text3', Pipeline([
            ('colext', TextSelector('lemmatized_review')),
            ('vect', CountVectorizer()),
        ])),
        ('words', Pipeline([
            ('wordext', NumberSelector('total_words')),
            ('wscaler', MinMaxScaler()),
        ])),
        # ('punct', Pipeline([
        #     ('colext', TextSelector('punctuation')),
        #     ('punct',PunctVectorizer())
        # ])),
        ('nouns', Pipeline([
            ('nounext', NumberSelector('num_np')),
            ('wscaler', MinMaxScaler()),
        ])),
        ('verbs', Pipeline([
            ('verbext', NumberSelector('num_vp')),
            ('wscaler', StandardScaler()),
        ])),
        ('modals', Pipeline([
                ('modalext', NumberSelector('num_md')),
            ('wscaler', StandardScaler()),
        ])),
])

# structural features
structural_feats = FeatureUnion([
        ('words', Pipeline([
            ('wordext', NumberSelector('total_words')),
            ('wscaler', MinMaxScaler()),
        ])),
])

# lexical features
lexical_feats = FeatureUnion([
        ('text2', Pipeline([
            ('colext', TextSelector('lemmatized_review')),
            ('tfidf', TfidfVectorizer(min_df=.0025, max_df=0.25, ngram_range=(1,3)))
        ])),
        ('text', Pipeline([
            ('colext', TextSelector('lemmatized_review')),
            ('vect', CustomVectorizer()),
            ('tfidf', TfidfTransformer()),
        ])),
        ('text3', Pipeline([
            ('colext', TextSelector('lemmatized_review')),
            ('vect', CountVectorizer()),
        ])),
        # original
        # ('text2', Pipeline([
        #     ('colext', TextSelector('lemmatized_review')),
        #     ('tfidf', TfidfVectorizer(min_df=.0025, max_df=0.25, ngram_range=(1,3)))
        # ])),
])

# punctuation features
punctuation_feats = FeatureUnion([
        ('text', Pipeline([
            ('colext', TextSelector('punctuation')),
            ('punct',PunctVectorizer())
        ])),
])

# contextual features
contextual_feats = FeatureUnion([
        ('nouns', Pipeline([
            ('nounext', NumberSelector('num_of_np')),
            ('wscaler', MinMaxScaler()),
        ])),
        ('verbs', Pipeline([
            ('verbext', NumberSelector('num_of_vp')),
            ('wscaler', MinMaxScaler()),
        ])),
        ('modals', Pipeline([
            ('modalext', NumberSelector('num_of_md')),
            ('wscaler', MinMaxScaler()),
        ])),
])

# lexical plus contextual
lexcon_feats = FeatureUnion([
        ('text', Pipeline([
            ('colext', TextSelector('text_final')),
            ('tfidf', TfidfVectorizer(min_df=.0025, max_df=0.25, ngram_range=(1,3)))
        ])),
        ('nouns', Pipeline([
            ('nounext', NumberSelector('num_of_np')),
            ('nscaler', MinMaxScaler()),
        ])),
        ('verbs', Pipeline([
            ('verbext', NumberSelector('num_of_vp')),
            ('vscaler', MinMaxScaler()),
        ])),
        ('modals', Pipeline([
            ('modalext', NumberSelector('num_of_md')),
            ('mscaler', MinMaxScaler()),
        ])),
])


# Classifier
print('defining classifier...')
rf_classifier = RandomForestClassifier(n_estimators=1000)
nb_classifier = naive_bayes.GaussianNB()
lr_classifier = LogisticRegression(solver ='liblinear')
svm_classifier = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')

pipe = Pipeline([('feats', lexical_feats),
                 ('clf', svm_classifier)
                 ])

print('training...')
pipe.fit(Train_X, Train_Y)
predicts = pipe.predict(Test_X)

print_stats(Test_Y, predicts, pipe.classes_)
print(confusion_matrix(Test_Y, predicts))
print(classification_report(Test_Y, predicts))
print('Hasil Task ', TASK)
print('Accuracy: ', accuracy_score(predicts,Test_Y)*100)
print('Precision = ', precision_score(Test_Y, predicts, average="macro"))
print('Recall score = ', recall_score(Test_Y, predicts, average="macro"))
print('F1 score = ', f1_score(Test_Y, predicts, average="macro"))

# end time
end = datetime.now()
print('total time = ', end - start)

# save the best model
filename = '../data/best_classifier_model.pk'
with open('filename','wb') as f:
    pickle.dump(pipe,f)


