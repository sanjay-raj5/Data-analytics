
# coding: utf-8

# In[102]:


import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
os.chdir("Z:/")


# In[103]:


def geo_mean(iterable):
    print(iterable)
    a = np.array(iterable)
    return a.prod()**(1.0/len(a))


# In[104]:


data = pd.read_csv("data.csv")


# In[105]:


data.drop(["Club Logo","Real Face","Photo"],axis=1,inplace=True)
data=data.iloc[:1000]


# In[106]:


nRow, nCol = data.shape
print(f'There are {nRow} rows and {nCol} columns')


# In[107]:


data.head()


# In[108]:


fig=plt.figure(figsize=(10,15))
ax=fig.gca()
data.hist(ax=ax,bins=20)
plt.show()


# In[109]:


bx_plt_data=data.head(100)
bx_plt_data.boxplot(column='Potential')
plt.show()


# In[110]:


data.mean()


# In[111]:


import math

def geomean(xs):
    return math.exp(math.fsum(math.log(x) for x in xs) / len(xs))

geomean(data.Potential)


# In[112]:


data.plot()


# In[113]:


data.iloc[:10].plot.box()
plt.show()


# In[114]:


data.iloc[:15].plot.bar()
plt.show()


# In[115]:


freq = {} 
for item in data.year: 
        if (item in freq): 
            freq[item] += 1
        else: 
            freq[item] = 1
key=[x for x in freq.keys()]
key
value=[freq[x] for x in freq.keys()]
value
plt.pie(value,labels=key,autopct='%.1f%%',startangle=90)
plt.show()
plt.bar(key,value)
plt.show()


# In[116]:


data.columns


# In[117]:


for i in data.columns:
    freq={}
    for item in data[i]: 
            if (item in freq): 
                freq[item] += 1
            else: 
                freq[item] = 1
    key=[x for x in freq.keys()]
    value=[freq[x] for x in freq.keys()]
    plt.pie(value,labels=key,autopct='%.1f%%',startangle=90)
    plt.plot()
    plt.show()
    


# In[ ]:


hmean=stats.hmean(pd.Potential)
print("hmean is",hmean)

