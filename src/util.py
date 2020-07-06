# -*- coding: utf-8 -*-
'''
author: akunaefi
email: akunaefi@st.cs.kumamoto-u.ac.jp

description:
    program util/library berisi functions multipurpose

'''

import pandas as pd
import nltk
from nltk import pos_tag, RegexpParser, Tree
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
import re
import string
import config

lemmatizer = WordNetLemmatizer()
stopword = list(set(stopwords.words('english')).difference(config.REMOVE_FROM_STOPWORDS))
stopword.extend(config.STOPWORDS_EXTEND)
indicators =  config.INDICATOR_LIST

def text_cleaner(sent):
    '''
    membersihkan tanda baca, dan mengoreksi singkatan (syntactical noise)
    '''
    sent = sent.lower()
    sent = re.sub(r"\'s", " is ", sent)
    # sent = re.sub(r"\'", "", sent)
    sent = re.sub(r"@", " ", sent)
    sent = re.sub(r"\'ve", " have ", sent)
    sent = re.sub(r"can't", "cannot ", sent)
    sent = re.sub(r"can not", "cannot ", sent)
    sent = re.sub(r"cant", "cannot ", sent)
    sent = re.sub(r"won't", "would not ", sent)
    sent = re.sub(r"n\'t", " not ", sent)
    sent = re.sub(r"i\'m", "i am ", sent)
    sent = re.sub(r"\'re", " are ", sent)
    sent = re.sub(r"\'d", " would ", sent)
    sent = re.sub(r"\'ll", " will ", sent)
    sent = re.sub(r"full review", " ", sent)
    sent = re.sub(r"(\d+)(k)", r"\g<1>000", sent)
    sent = re.sub('[%s]' % re.escape(string.punctuation), ' ', sent)
    sent = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", sent)
    return sent


def punctuation_extractor(sent):
    
    # meng-extract punctuation dari sentence
    punc_list = ['!', '"', '#', '$', '%', '&', '\'' ,'(' ,')', '*', '+', ',', '-', '.' ,'/' ,':' ,';' ,'' ,'?' ,'@' ,'[' ,'\\' ,']' ,'^' ,'_' ,'`' ,'{' ,'|' ,'}' ,'~']
    sent = sent.replace("\\r\\n"," ")
    for character in sent:
        if character not in punc_list:
            sent = sent.replace(character, "")
    return sent


def nltk2wn_tag(nltk_tag):
  if nltk_tag.startswith('J'):
    return wordnet.ADJ
  elif nltk_tag.startswith('V'):
    return wordnet.VERB
  elif nltk_tag.startswith('N'):
    return wordnet.NOUN
  elif nltk_tag.startswith('R'):
    return wordnet.ADV
  else:                    
    return None


def lemmatize_sentence(sentence):
    nltk_tagged = pos_tag(word_tokenize(sentence))    
    wn_tagged = map(lambda x: (x[0], nltk2wn_tag(x[1])), nltk_tagged)
    res_words = []
    for word, tag in wn_tagged:
        if word not in stopword and word.isalpha():
            if tag is None:                        
                res_words.append(word)
            else:
                res_words.append(lemmatizer.lemmatize(word, tag))
    return " ".join(res_words)


def check_grammar(sent):
    words = word_tokenize(sent)
    tagged = pos_tag(words)
    chunkGrammar = "ARG: {<VP><VBG|NP>}"
    chunkParser = RegexpParser(chunkGrammar)
    chunked = chunkParser.parse(tagged)
    num = 0
    for child in chunked:
        # num = 0
        if isinstance(child, Tree):               
            if child.label() == 'ARG':
                num += 1
                # print(sent)
                # print(num)
    return num    

def check_indicator(sent):
    check = 0
    for item in indicators:
        if item in sent:
            # print(item)
            check = 1
            break
    
    return check

    
    
def text_preprocess(df):
    '''
    pro-processing data frame yang mengandung text
    terdiri dari tokenization, lemmatization
    
    input: df yang didalamnya ada kolom 'review'
    '''
    
    print('lemmatizing...')
    for index, entry in enumerate(df['review']):
        puncts = punctuation_extractor(entry)
        sent = text_cleaner(entry)
        total_words = len(entry)
        grammar = check_grammar(sent)
        indicator = check_indicator(sent)
        df.loc[index,'cleaned_review'] = sent
        df.loc[index,'grammar'] = grammar
        df.loc[index,'indicator'] = indicator
        df.loc[index,'lemmatized_review'] = lemmatize_sentence(sent)
        df.loc[index,'total_words'] = total_words # for structural feature
        df.loc[index,'punctuation'] = puncts

    return df.copy()

# a function to traverse a phrase to find the constituents (leaves)
# it is written in recursive way, can be changed into a loop
def traverse_phrase(tree):
    for subtree in tree:
        if type(subtree) == nltk.tree.Tree:
            traverse_phrase(subtree)
        # else:
        #     print( "constituent : " + subtree )

# a function to traverse a tree to find NP
# fungsi untuk menghitung jumlah clause/frase, apakah NP, VP, PP, AP
def traverse_tree(tree, phrase_type):
    my_frase = []
    for subtree in tree:
        if type(subtree) == nltk.tree.Tree:
            if subtree.label() == phrase_type:
                # print("\n[Verb Phrase]")
                traverse_phrase(subtree)
                my_frase.append(subtree.copy(True))
            else:
                list_of_ = traverse_tree(subtree,phrase_type)
                my_frase.extend(list_of_)
    return my_frase


if __name__ == '__main__':
    
    # cetak kalimat yang lebih dari 5 kata
    df = pd.read_csv('../data/reviews_all.csv')
    df_preprocessed = text_preprocess(df)
    df_ori = pd.read_pickle('../data/feature_extraction_decision_sample.pkl')
    df = df_ori.filter(['lemmatized_review','category','decision_category','total_words','punctuation','num_of_np','num_of_vp','num_of_md'],axis=1)
    df.fillna(0,inplace=True)
    print(df.columns.tolist())

    # for index, line in enumerate(df_ori['lemmatized_review']):
    #     if len(line)>5:
    #         print(line)
