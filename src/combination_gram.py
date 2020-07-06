# -*- coding: utf-8 -*-
"""
Created on Tue May  5 20:39:37 2020

@author: Anang Kunaefi
"""

# percobaan combination gram

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from itertools import permutations, combinations

text1 = 'the quick brown fox payment pay'

stop_list = stopwords.words('english')
word_list = [word for word in word_tokenize(text1) if word not in stop_list]

# generate all permutations
pgram_list = list(permutations(word_list,2))
cgram_list = list(combinations(word_list,2))

for entry in pgram_list:
    print(" ".join(entry))

# since too many entries, should eliminate few
# alternative: wordnet, word2vec,
i = 1 

new_cgram_list=[]
for entry in pgram_list:
    w1 = wordnet.synsets(entry[0])
    w2 = wordnet.synsets(entry[1])
    if(len(w1)!=0 and len(w2)!=0):
        sim = w1[0].wup_similarity(w2[0])
        sim_path = w1[0].path_similarity(w2[0]) # path sim is not relevant, keep with wup
        
        if sim < 0.7:
            cgram_words = " ".join(entry)
            new_cgram_list.append(cgram_words)
        else:
            print(f'{w1[0]} - {w2[0]} = {sim}')
    # print('word1=',entry[0])
    # print('word2=',entry[1])
    # print(f'{i}---sim = {sim}')
    i += 1
print(new_cgram_list)

