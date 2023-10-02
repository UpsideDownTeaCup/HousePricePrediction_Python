#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from matplotlib import pyplot
import xgboost
from xgboost import XGBRegressor
import pickle
import sys
import scipy
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import csv


# In[3]:


df = pd.read_csv("HousePriceDataTest.csv")


# In[6]:


df['Date'] = df.Date.str[6:]
df['Date'] = df['Date'].astype(int)


# In[7]:


df.describe()


# In[8]:


with open("modelxgbR","rb") as f:
    pickle.load(xgbR,f)


# In[ ]:




