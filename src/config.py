# -*- coding: utf-8 -*-
'''
author: akunaefi@st.cs.kumamoto-u.ac.jp

description:
    file konfigurasi

'''

import nltk
from nltk.corpus import stopwords

IS_USING_STOPWORD = False

NUM_OF_SAMPLE = 100

# in stab's paper indicator is a list of argumentative keywords or
# in other term discourse marker
INDICATOR_LIST = ['therefore','thus','consequently','because',
                  'reason','furthermore','so that','so','actually',
                  'basically','however','nevertheless','alternatively',
                  'though','otherwise','instead','nonetheless',
                  'conversely','similarly','comparable','likewise',
                  'further','moreover','addition','additionally',
                  'then','besides','hence','therefore','accordingly',
                  'consequently','thereupon','as a result','since',
                  'whenever']

VERB_LIST = ['believe','think','agree']

ADVERB_LIST = ['also','often','really']

MODAL_LIST = ['should','could','would','might','must', 'can',
              'may','shall','will','ought to','need','have to',
              'used to']

STOPWORDS_EXTEND = ['app','apps','review','adobe','avlpro','bloomberg','fitbit','foursquared',
                 'gopro','kahoot','payoneer','smule','supermario','polaris','office',
                  'thanks','best','good','song','game',
                  'mario','great','phone','awesome','nintendo','wattpad','weather',
                  'really','okay','ever','photoshop','foursquare','underground','fitbit']

REMOVE_FROM_STOPWORDS = ['because','further','should','will']

GRAMMAR_LIST = ['']

if __name__ == '__main__':
    
    stop = set(stopwords.words('english')).difference(REMOVE_FROM_STOPWORDS)
    
    print(stop)


