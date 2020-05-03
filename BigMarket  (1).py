#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# reading train and test data
train_data = pd.read_csv('Train.csv')
test_data = pd.read_csv('test.csv')


# In[3]:


train_data.head()


# In[4]:


df = pd.read_csv("train.csv") 
df.shape


# In[5]:


df.isnull().any()


# In[6]:


df.info()


# In[ ]:





# In[7]:



df['Item_MRP'].hist(bins=10) 
 
# shows presence of a lot of outliers/extreme values 
df.boxplot(column='Item_MRP', by = 'Outlet_Establishment_Year') 


# In[8]:


df['Item_Weight'].plot.kde()


# In[9]:


# input 
x = train_data.iloc[:, [3, 5]].values 

# output 
y = train_data.iloc[:, 4].values 


# In[10]:


from sklearn.model_selection import train_test_split 
xtrain, xtest, ytrain, ytest = train_test_split( x, y, test_size = 0.25, random_state = 0) 


# In[11]:



from sklearn.preprocessing import StandardScaler 
sc_x = StandardScaler() 
xtrain = sc_x.fit_transform(xtrain)  
xtest = sc_x.transform(xtest) 
  
print (xtrain[0:10, :]) 


# In[12]:


from sklearn.linear_model import LogisticRegression 
classifier = LogisticRegression(random_state = 0) 
classifier.fit(xtrain, ytrain) 


# In[13]:


y_pred = classifier.predict(xtest) 


# In[14]:


from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(ytest, y_pred) 

print ("Confusion Matrix : \n", cm) 


# In[15]:


from sklearn.metrics import accuracy_score 
print ("Accuracy : ", accuracy_score(ytest, y_pred)) 


# In[ ]:




