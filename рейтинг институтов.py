#!/usr/bin/env python
# coding: utf-8

# In[108]:


import os
os.chdir("C:\python")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DF = pd.read_csv('awesome_calc.csv', encoding ='windows-1251', sep=";", header = 0, index_col=0)
DF1 = pd.read_csv('awesome_calc2.csv', encoding ='windows-1251', sep=";", header = 0, index_col=0)
#DF=DF.drop(['Arts & Humanities','Engineering & Technology','Life Sciences & Medicine','Natural Sciences','Social Sciences & Management'])
#DF.dropna(inplace = True)
#df.drop(['Cochice', 'Pima'])


# In[109]:


DF


# In[110]:


DF1


# In[111]:


DF['1.1'] = DF['1.1'].str.replace(',','.').astype(float)
DF['1.2'] = DF['1.2'].str.replace(',','.').astype(float)
DF['1.4'] = DF['1.4'].str.replace(',','.').astype(float)
DF['1.5'] = DF['1.5'].str.replace(',','.').astype(float)


# In[118]:


stud2020  = DF[['1.1','1.2','1.4','1.5']]
stud2020
ax = stud2020.sort_values(by=['1.1']).plot.line(figsize=(10,5), rot = 90, subplots=True)
#ax.set_yscale('log')
DF[['1.1','1.2','1.4']].astype(float)


# In[ ]:




