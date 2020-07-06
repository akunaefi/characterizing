# -*- coding: utf-8 -*-
"""
Created on Sat May  2 18:41:07 2020

@author: Anang Kunaefi

description: 
    code untuk extract feature dari text
    
"""

import os
import pandas as pd
import nltk
from nltk.parse.stanford import StanfordParser
import util
import pickle
import categorize_reviews as cr
import config

# a function to traverse a phrase to find the constituents (leaves)
# it is written in recursive way, can be changed into a loop
def traverse_phrase(tree):
    for subtree in tree:
        if type(subtree) == nltk.tree.Tree:
            traverse_phrase(subtree)
        # else:
        #     print( "constituent : " + subtree )

# a function to traverse a tree to find NP
# it is written in recursive way, can be changed into a loop
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


def get_parse_tree(df):
    print('\ntraversing phrase...')

    # path setting for stanford parser
    # versi windows
    # java_path = r'C:\Program Files\Java\jdk1.8.0_151\bin'
    # os.environ['JAVAHOME'] = java_path
    # stanford_parser = StanfordParser(path_to_jar='c:/stanford-parser-full/stanford-parser.jar',
    #                       path_to_models_jar='c:/stanford-parser-full/stanford-parser-3.5.2-models.jar')
    
    # versi linux
    java_path = r'/usr/lib/jvm/java-1.8.0-openjdk-amd64/bin'
    os.environ['JAVAHOME'] = java_path
    stanford_parser = StanfordParser(path_to_jar='/home/akunaefi/PycharmProjects/StanfordParser/stanford-parser-full-2015-04-20/stanford-parser.jar',
                         path_to_models_jar='/home/akunaefi/PycharmProjects/StanfordParser/stanford-parser-full-2015-04-20/stanford-parser-3.5.2-models.jar')

    for index,words in enumerate(df['lemmatized_review']):
        # list_of_frases = []
        print('--------------------- ', index)
        print('review: ', words)
        # sent = text_cleaner(words)
        # df_clause.loc[index,'Review'] = sent
        parsed_sent = stanford_parser.raw_parse(words)
        tree = parsed_sent.__next__()
        list_of_nps = traverse_tree(tree, 'NP')
        num_of_np = len(list_of_nps)
        print('NOUN = ', num_of_np)
        list_of_vps = traverse_tree(tree, 'VP')
        num_of_vp = len(list_of_vps)
        print('VERB = ', num_of_vp)
        list_of_mds = traverse_tree(tree, 'MD')
        num_of_md = len(list_of_mds)
        print('MODAL = ', num_of_md)
        df.loc[index,'num_np'] = num_of_np
        df.loc[index,'num_vp'] = num_of_vp
        df.loc[index, 'num_md'] = num_of_md

    return df.copy()

    
if __name__ == '__main__':
    
    # ambil dataset
    # df = pd.read_csv('../data/reviews_all.csv')

    # for task 1
    # df_t1 = cr.get_dataset_task1(df)
    # df_t1_preprocessed = util.text_preprocess(df_t1)
    # null_index = df_t1_preprocessed[df_t1_preprocessed['lemmatized_review'] == ""].index 
    # df_t1_preprocessed.drop(null_index,inplace=True) 
    # df_t1_preprocessed = get_parse_tree(df_t1_preprocessed)
    # print('\nsaving features for task 1...')
    # with open('../data/feature_extraction_task1.pkl', 'wb') as f1:
    #     pickle.dump(df_t1_preprocessed, f1)
    
    
    # for task 2
    # df_t2 = cr.get_dataset_task2(df)
    # df_t2_preprocessed = util.text_preprocess(df_t2)
    # null_index = df_t2_preprocessed[df_t2_preprocessed['lemmatized_review'] == ""].index 
    # df_t2_preprocessed.drop(null_index,inplace=True) 
    # df_t2_preprocessed = get_parse_tree(df_t2_preprocessed)
    # print('\nsaving features for task 2...')
    # with open('../data/feature_extraction_task2.pkl', 'wb') as f2:
    #     pickle.dump(df_t2_preprocessed, f2)


    # for task 3
    # df_t3 = cr.get_dataset_task3(config.NUM_OF_SAMPLE)  
    # df_t3_preprocessed = util.text_preprocess(df_t3)
    # null_index = df_t3_preprocessed[df_t3_preprocessed['lemmatized_review'] == ""].index 
    # df_t3_preprocessed.drop(null_index,inplace=True) 
    # df_t3_preprocessed = get_parse_tree(df_t3_preprocessed)
    # print('\nsaving features for task 3...')
    # with open('../data/feature_extraction_task3.pkl', 'wb') as f3:
    #     pickle.dump(df_t3_preprocessed, f3)
    # print('done.')

    # fit feature extraction
    dataset = cr.get_fitbit()
    # dataset = dataset.filter(['review','decision_category'],axis=1)
    dataset.drop(dataset[dataset.category < 1].index, inplace=True)
    print(dataset.shape)
    
    # separate the decision class
    acquire = dataset['decision_category'] == 0
    df_acquire = dataset[acquire].copy()
    df_acquire['decision_category'] = 'acquire'
    # print(df_acquire.head())
    buying = dataset['decision_category'] == 1
    df_buying = dataset[buying].copy()
    df_buying['decision_category'] = 'buying'
    recommend = dataset['decision_category'] == 2
    df_recommend = dataset[recommend].copy()
    df_recommend['decision_category'] = 'recommend'
    request = dataset['decision_category'] == 3
    df_request = dataset[request].copy()
    df_request['decision_category'] = 'request'
    rating = dataset['decision_category'] == 4
    df_rating = dataset[rating].copy()
    df_rating['decision_category'] = 'rating'
    
    
    # masing2 25
    df_buying = df_buying.append(df_buying.sample(n=8), ignore_index = True)
    df_request = df_request.append(df_request.sample(n=10), ignore_index = True)
    df_rating = df_rating.append(df_rating.sample(n=2), ignore_index = True)
    df_acquire = df_acquire.sample(n=25)
    df_recommend = df_recommend.sample(n=25)
    
    df_fitbit = pd.concat([df_acquire, df_buying, df_recommend, df_request, df_rating],sort=True).reset_index(drop=True)
    print('data shape after sampling', df_fitbit.shape)
    
    # preprocessing
    # df_fitbit_preprocessed = util.text_preprocess(df_fitbit)
    # null_index = df_fitbit_preprocessed[df_fitbit_preprocessed['lemmatized_review'] == ""].index 
    # df_fitbit_preprocessed.drop(null_index,inplace=True) 
    # df_fitbit_preprocessed = get_parse_tree(df_fitbit_preprocessed)
    # print('\nsaving fitbit features for task 3...')
    # with open('../data/feature_extraction_fitbit.pkl', 'wb') as f3:
    #     pickle.dump(df_fitbit_preprocessed, f3)
    # print('done.')    
   
