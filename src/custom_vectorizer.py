# -*- coding: utf-8 -*-
"""
Created on Wed May  6 10:11:37 2020

@author: Anang Kunaefi
"""

import pandas as pd
from itertools import permutations, combinations
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet


# defines a custom vectorizer class
class CustomVectorizer(CountVectorizer): 
    
    # overwrite the build_analyzer method, allowing one to
    # create a custom analyzer for the vectorizer
    def build_analyzer(self):
        
        # load stop words using CountVectorizer's built in method
        stop_words = self.get_stop_words()
        
        # create the analyzer that will be returned by this method
        def analyser(doc):
            
            # if len(doc) != 0:
            tokens = [word for word in word_tokenize(doc)]
                        
            # generate permutation words
            new_tokens_list = list(permutations(tokens, 2)) 
            
            
            # presicion lebih bagus, tapi recall masih jelek
            # coba pake semantic similarity 
            for entry in new_tokens_list:
                w1 = wordnet.synsets(entry[0])
                w2 = wordnet.synsets(entry[1])
                if(len(w1)!=0 and len(w2)!=0):
                    sim = w1[0].wup_similarity(w2[0])
                    if sim is not None:
                        if sim <= 0.8:
                            tokens.append(" ".join(entry))
                        # else:
                            # print(f'{w1[0]} - {w2[0]} = {sim}')    
            
            return tokens
        
        return(analyser)

class PunctVectorizer(CountVectorizer):
    
    def prepare_doc(self, doc):
        punc_list = ['!', '"', '#', '$', '%', '&', '\'' ,'(' ,')', '*', '+', ',', '-', '.' ,'/' ,':' ,';' ,'' ,'?' ,'@' ,'[' ,'\\' ,']' ,'^' ,'_' ,'`' ,'{' ,'|' ,'}' ,'~']
        doc = doc.replace("\\r\\n"," ")
        for character in doc:
            if character not in punc_list:
                doc = doc.replace(character, "")
        return doc

    def build_analyzer(self):
        preprocess = self.build_preprocessor()
        return lambda doc : preprocess(self.decode(self.prepare_doc(doc)))
    
    
    
if __name__ == '__main__':
    corpora = ['the quick brown payment price', 'jump over the lazy dog']
    
    # permutation
    custom_vec = CustomVectorizer(stop_words='english')
    word_matrix = custom_vec.fit_transform(corpora)
    tokens = custom_vec.get_feature_names()
    
    # punct
    # punct_vec = PunctVectorizer()
    # punct_matrix = punct_vec.fit_transform(corpora)
    # puncts = punct_vec.get_feature_names()
    
    df = pd.DataFrame(data=word_matrix.toarray(), index=['doc1','doc2'], columns=tokens)
    # df = pd.DataFrame(data=punct_matrix.toarray(), index=['doc1','doc2'], columns=puncts)
    print(df)

