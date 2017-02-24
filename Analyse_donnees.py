# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 10:00:46 2017

@author: May-line
"""

import pandas as pd

#train=pd.read_csv('train_users_2.csv')
#test=pd.read_csv('test_users.csv')

#train.values[:,15].value_counts()
histo=pd.value_counts(train.values[:,15])
print(histo)
histo[1:].plot.bar()
