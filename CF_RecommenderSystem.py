#!/usr/bin/env python
# coding: utf-8

# # Recommender Systems
# ## Writing an Algorithm for Recommending Movies with a Collaborative Filtering
# In dit script probeer ik een algoritme te schrijven die de beoordeling van films gaat voorspellen voor mensen. Recommender Systems worden gebruikt door bedrijven zoals YouTube en Netflix om jou langer op hun platform te houden. Ook kan het gebruikt worden door online kledingwinkels die willen voorspellen welke kleding jij leuk vindt.
# 
# De dataset die wordt gebruikt is de MovieLens database. Deze is te vinden op [grouplens.org/](https://grouplens.org/datasets/movielens/). Daarnaast gebruik ik de kennis die ik heb opgedaan vanuit dit [Towards Data Science](https://towardsdatascience.com/building-and-testing-recommender-systems-with-surprise-step-by-step-d4ba702ef80b) artikel.
# 
# Deze notebook is een gevolg op `CB_RecommenderSystem.ipynb` en mijn aanbeveling is om die eerst te snappen. Die geeft je in het simpel het idee van een recommender system. In tegenstelling tot CB probeert Collaborative Filtering te kijken naar de patronen van de gebruikers. 

# In[1]:


import pandas as pd 
import numpy as np
import warnings
import matplotlib.pyplot as plt

from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split
from surprise import accuracy

get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')


# Het belangrijkste wat in de volgende cell gebeurd is het inlezen van de data en het samenvoegen van de twee datasets. De variable `BigDataSet` bepaald of je de grote data set pakt van 20 miljoen regels (`True`) of de kleine dataset van 100.000 regels (`False`).

# ## Data Preprocessing

# Voor de surprise methode is er bijna geen data preprocessing nodig. Het enige wat nodig is is een tabel met userID's, movieID's en ratings.
# 
# | userID | movieID | Rating |
# | --- | --- | --- |
# | 1 | 1 | 5 |
# | 1 | 2 | 3 |
# | 2 | 1 | 4 |
# 

# In[2]:


BigDataSet = True
DataSet = "data" if BigDataSet else "data_small"

df = pd.read_csv(DataSet + '/ratings.csv', sep=',', names=['userID','movieID','rating','timestamp'], header = 0)
df.head()


# ## Scikit surprise approach
# https://towardsdatascience.com/building-and-testing-recommender-systems-with-surprise-step-by-step-d4ba702ef80b

# In[ ]:


# Train Test split fit
reader = Reader(rating_scale=(0, 5))
data = Dataset.load_from_df(df[["userID", "movieID", "rating"]], reader)
trainSet, testSet = train_test_split(data, test_size = 0.2)

movieSVD = SVD(random_state = 1, verbose = True, n_epochs = 5)
movieSVD.fit(trainSet)

# test the algorithm
predictions = movieSVD.test(testSet)
accuracy.rmse(predictions)


# In[36]:


def get_Iu(uid):
    """ return the number of items rated by given user
    args: 
      uid: the id of the user
    returns: 
      the number of items rated by the user
    """
    try:
        return len(trainSet.ur[trainSet.to_inner_uid(uid)])
    except ValueError: # user was not part of the trainset
        return 0
    
def get_Ui(iid):
    """ return number of users that have rated given item
    args:
      iid: the raw id of the item
    returns:
      the number of users that have rated the item.
    """
    try: 
        return len(trainSet.ir[trainSet.to_inner_iid(iid)])
    except ValueError:
        return 0


# In[37]:


dfPred = pd.DataFrame(predictions, columns=['uid', 'iid', 'rui', 'est', 'details'])
dfPred = dfPred.drop('details', axis = 1)
dfPred["SE"] = (dfPred.est-dfPred.rui)**2
dfPred['Iu'] = dfPred.uid.apply(get_Iu)
dfPred['Ui'] = dfPred.iid.apply(get_Ui)
dfPred = dfPred.sort_values(by = "SE", ascending = False)
dfPred.head(30)


# In[41]:


dfPred.SE.hist()

